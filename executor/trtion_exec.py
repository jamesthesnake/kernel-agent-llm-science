# kernel-agent-llm-science/kernel_agent/executor/triton_exec.py
from __future__ import annotations
from typing import Dict, Any, List
import torch
import triton
import triton.language as tl
from .common import time_ms, softmax_reference, elem_size_bytes, dtype_to_torch
from kernel_agent.schemas import TritonPlan, Results, ConfigResult

def _load_kernel_from_string(src: str):
    # Execute the kernel string in a sandboxed namespace exposing triton, tl, torch
    ns = {}
    exec(src, {"triton": triton, "tl": tl, "torch": torch}, ns)
    # Find any function whose name ends with '_kernel'
    candidates = [v for k, v in ns.items() if k.endswith("_kernel")]
    if not candidates:
        raise ValueError("No *_kernel function found in provided Triton source.")
    return candidates[0]

def _bytes_softmax(B: int, N: int, dtype: str) -> int:
    # lower-bound: read+write once
    return 2 * B * N * elem_size_bytes(dtype)

@torch.inference_mode()
def run(plan: TritonPlan) -> Results:
    kernel = _load_kernel_from_string(plan.triton_kernel)
    torch.cuda.synchronize()
    tested: List[ConfigResult] = []
    dtype = plan.dtype
    torch_dtype = dtype_to_torch(dtype)

    for shape in plan.shapes:
        B, N = shape.B, shape.N
        x = torch.randn((B, N), device="cuda", dtype=torch_dtype)
        ref = softmax_reference(x, torch_dtype)
        # Sweep param grid
        for BLOCK in plan.param_grid.get("BLOCK", [128]):
            for num_warps in plan.param_grid.get("num_warps", [4]):
                for num_stages in plan.param_grid.get("num_stages", [2]):
                    # Launch: one program per row
                    grid = (B,)
                    def _call():
                        kernel[grid](
                            x, x,  # in-place out; executor compares to ref computed before
                            B=B, N=N,
                            BLOCK=BLOCK,
                            num_warps=num_warps,
                            num_stages=num_stages
                        )
                    # Time
                    try:
                        lat = time_ms(_call, iters=50, warmup=10)
                        torch.cuda.synchronize()
                        # Grab output (kernel wrote into x)
                        out = x
                        linf = (out - ref).abs().max().item()
                        tol = plan.tolerance["row_softmax"][plan.dtype]
                        passed = linf <= tol
                        tput = _bytes_softmax(B, N, dtype) / (lat * 1e-3) / 1e9
                        tested.append({
                            "config": {"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages},
                            "shape": {"B": B, "N": N},
                            "latency_ms": float(lat),
                            "throughput_gbps": float(tput),
                            "achieved_occupancy": None,  # not measured; needs CUPTI/Nsight
                            "l_inf_error": float(linf),
                            "passed": bool(passed),
                        })
                    except Exception as e:
                        tested.append({
                            "config": {"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages},
                            "shape": {"B": B, "N": N},
                            "latency_ms": float("inf"),
                            "throughput_gbps": None,
                            "achieved_occupancy": None,
                            "l_inf_error": float("inf"),
                            "passed": False,
                            "notes": f"exception: {type(e).__name__}: {e}",
                        })
        # free
        del x, ref
    # pick best among passing
    passing = [r for r in tested if r.get("passed")]
    best = None
    if passing:
        best = min(passing, key=lambda r: r["latency_ms"])
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
        executor_info={}
    )
