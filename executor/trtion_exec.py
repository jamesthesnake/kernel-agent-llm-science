from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
import triton
import triton.language as tl
from .common import time_ms, softmax_reference, elem_size_bytes, dtype_to_torch, max_ulp_error, apply_vram_quota_gb
from kernel_agent.schemas import TritonPlan, Results, ConfigResult

def _load_kernel_from_string(src: str):
    ns = {}
    exec(src, {"triton": triton, "tl": tl, "torch": torch}, ns)
    candidates = [v for k, v in ns.items() if k.endswith("_kernel")]
    if not candidates:
        raise ValueError("No *_kernel function found in provided Triton source.")
    return candidates[0]

def _bytes_softmax_three_pass(B: int, N: int, dtype: str) -> int:
    # 3 passes (read + read + read/write) ~ 3 * N elements per row
    return 3 * B * N * elem_size_bytes(dtype)

@torch.inference_mode()
def run(plan: TritonPlan, *, device: int = 0, timeout_s: Optional[float] = None,
        vram_gb: Optional[float] = None, verify_device: Optional[int] = None) -> Results:
    torch.cuda.set_device(device)
    apply_vram_quota_gb(vram_gb, device)

    kernel = _load_kernel_from_string(plan.triton_kernel)
    tested: List[ConfigResult] = []
    torch_dtype = dtype_to_torch(plan.dtype)

    for shape in plan.shapes:
        B, N = shape.B, shape.N
        x = torch.randn((B, N), device=f"cuda:{device}", dtype=torch_dtype)
        y = torch.empty_like(x)
        ref = softmax_reference(x, torch_dtype)

        for BLOCK in plan.param_grid.get("BLOCK", [128]):
            for num_warps in plan.param_grid.get("num_warps", [4]):
                for num_stages in plan.param_grid.get("num_stages", [2]):
                    grid = (B,)
                    def _call():
                        kernel[grid](
                            x, y,
                            B=B, N=N,
                            BLOCK=BLOCK,
                            num_warps=num_warps,
                            num_stages=num_stages
                        )
                    try:
                        lat = time_ms(_call, iters=50, warmup=10, max_seconds=timeout_s)
                        torch.cuda.synchronize(device)
                        out = y
                        linf = (out - ref).abs().max().item()
                        ulp = max_ulp_error(out, ref, torch_dtype)
                        tol = plan.tolerance["row_softmax"][plan.dtype]
                        passed = (linf <= tol)
                        tput = _bytes_softmax_three_pass(B, N, plan.dtype) / (lat * 1e-3) / 1e9
                        tested.append({
                            "config": {"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages},
                            "shape": {"B": B, "N": N},
                            "latency_ms": float(lat),
                            "throughput_gbps": float(tput),
                            "achieved_occupancy": None,
                            "l_inf_error": float(linf),
                            "ulp_error": float(ulp),
                            "passed": bool(passed),
                        })
                    except TimeoutError as e:
                        tested.append({
                            "config": {"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages},
                            "shape": {"B": B, "N": N},
                            "latency_ms": float("inf"),
                            "throughput_gbps": None,
                            "achieved_occupancy": None,
                            "l_inf_error": float("inf"),
                            "ulp_error": None,
                            "passed": False,
                            "notes": f"timeout: {e}",
                        })
                    except Exception as e:
                        tested.append({
                            "config": {"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages},
                            "shape": {"B": B, "N": N},
                            "latency_ms": float("inf"),
                            "throughput_gbps": None,
                            "achieved_occupancy": None,
                            "l_inf_error": float("inf"),
                            "ulp_error": None,
                            "passed": False,
                            "notes": f"exception: {type(e).__name__}: {e}",
                        })
        del x, y, ref

    passing = [r for r in tested if r.get("passed")]
    best = min(passing, key=lambda r: r["latency_ms"]) if passing else None

    # Optional second-GPU re-verification on the best config
    recheck = None
    if best and verify_device is not None and verify_device != device and verify_device < torch.cuda.device_count():
        torch.cuda.synchronize(device)
        torch.cuda.set_device(verify_device)
        apply_vram_quota_gb(vram_gb, verify_device)
        shape = best["shape"]; B, N = shape["B"], shape["N"]
        conf = best["config"]
        x2 = torch.randn((B, N), device=f"cuda:{verify_device}", dtype=torch_dtype)
        y2 = torch.empty_like(x2)
        ref2 = softmax_reference(x2, torch_dtype)
        grid = (B,)
        def _call2():
            kernel[grid](x2, y2, B=B, N=N, BLOCK=conf["BLOCK"],
                         num_warps=conf["num_warps"], num_stages=conf["num_stages"])
        try:
            lat2 = time_ms(_call2, iters=20, warmup=5, max_seconds=timeout_s)
            linf2 = (y2 - ref2).abs().max().item()
            ulp2 = max_ulp_error(y2, ref2, torch_dtype)
            recheck = {
                "device": int(verify_device),
                "latency_ms": float(lat2),
                "l_inf_error": float(linf2),
                "ulp_error": float(ulp2),
                "passed": bool(linf2 <= plan.tolerance["row_softmax"][plan.dtype])
            }
        except Exception as e:
            recheck = {"device": int(verify_device), "notes": f"recheck_failed: {e}", "passed": False}

        # restore primary device
        torch.cuda.set_device(device)

    return Results(
        experiment_id=plan.experiment_id,
        backend="triton",
        op=plan.op,
        dtype=plan.dtype,
        shapes=[s.model_dump() for s in plan.shapes],
        hypothesis=plan.hypothesis,
        tolerance=plan.tolerance,
        tested=tested,
        best=best,
        executor_info={},
        recheck=recheck
    )
