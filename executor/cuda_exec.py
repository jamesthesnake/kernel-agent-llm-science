from __future__ import annotations
from typing import List, Optional
import math
import torch
from torch.utils.cpp_extension import load_inline
from .common import (
    time_ms, stencil3x3_reference, elem_size_bytes, dtype_to_torch,
    max_ulp_error, apply_vram_quota_gb, time_eager_stencil_ms
)
from agents.schemas import CudaPlan, Results, ConfigResult

WRAPPER_TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
extern "C" __global__ void stencil3x3_kernel(const {SCALAR}* __restrict__ inp,
                                             {SCALAR}* __restrict__ out,
                                             int H, int W);

static void launch_kernel(torch::Tensor inp, torch::Tensor out, int bx, int by) {
  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  auto H = (int)inp.size(0);
  auto W = (int)inp.size(1);
  dim3 block(bx, by, 1);
  dim3 grid((W + bx - 1) / bx, (H + by - 1) / by, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  stencil3x3_kernel<<<grid, block, 0, stream>>>(
      ({SCALAR}*)inp.data_ptr<{SCALAR}>(),
      ({SCALAR}*)out.data_ptr<{SCALAR}>(),
      H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch_kernel, "launch");
}
"""

def _compile_module(kernel_src: str, scalar: str, name: str):
    full_src = kernel_src + "\n" + WRAPPER_TEMPLATE.replace("{SCALAR}", scalar)
    extra_cuda_cflags = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-gencode=arch=compute_90,code=sm_90"]
    mod = load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=full_src,
        functions=["launch"],
        with_cuda=True,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    return mod

@torch.inference_mode()
def run(plan: CudaPlan, *, device: int = 0, timeout_s: Optional[float] = None,
        vram_gb: Optional[float] = None, verify_device: Optional[int] = None) -> Results:
    torch.cuda.set_device(device)
    apply_vram_quota_gb(vram_gb, device)

    if plan.dtype != "fp32":
        raise NotImplementedError("Round 0 CUDA path implemented for fp32 only.")
    scalar = "float"
    mod = _compile_module(plan.cuda_kernel, scalar, name=f"stencil3x3_{plan.experiment_id}")
    torch_dtype = dtype_to_torch(plan.dtype)
    tested: List[ConfigResult] = []

    for shape in plan.shapes:
        H, W = shape.H, shape.W
        inp = torch.randn((H, W), device=f"cuda:{device}", dtype=torch_dtype)
        ref = stencil3x3_reference(inp.to(torch.float32)).to(torch_dtype)
        out = torch.empty_like(inp)
        bytes_moved = (9 + 1) * H * W * elem_size_bytes(plan.dtype)

        # Eager baseline once per shape
        baseline_ms = time_eager_stencil_ms(inp, iters=plan.iters, warmup=max(5, plan.iters // 5), timeout_s=timeout_s)

        for bx in plan.param_grid.get("BLOCK_X", [16]):
            for by in plan.param_grid.get("BLOCK_Y", [16]):
                def _call():
                    mod.launch(inp, out, int(bx), int(by))
                try:
                    lat = time_ms(_call, iters=plan.iters, warmup=max(5, plan.iters // 5), max_seconds=timeout_s)
                    linf = (out - ref).abs().max().item()
                    ulp = max_ulp_error(out, ref, torch_dtype)
                    tol = plan.tolerance["stencil3x3"][plan.dtype]
                    passed = linf <= tol
                    tput = bytes_moved / (lat * 1e-3) / 1e9
                    speedup = float(baseline_ms / lat) if (baseline_ms and math.isfinite(lat)) else None
                    tested.append({
                        "config": {"BLOCK_X": bx, "BLOCK_Y": by},
                        "shape": {"H": H, "W": W},
                        "latency_ms": float(lat),
                        "throughput_gbps": float(tput),
                        "achieved_occupancy": None,
                        "l_inf_error": float(linf),
                        "ulp_error": float(ulp),
                        "baseline_latency_ms": float(baseline_ms),
                        "speedup_vs_baseline": speedup,
                        "passed": bool(passed),
                    })
                except TimeoutError as e:
                    tested.append({
                        "config": {"BLOCK_X": bx, "BLOCK_Y": by},
                        "shape": {"H": H, "W": W},
                        "latency_ms": float("inf"),
                        "throughput_gbps": None,
                        "achieved_occupancy": None,
                        "l_inf_error": float("inf"),
                        "ulp_error": None,
                        "baseline_latency_ms": float(baseline_ms),
                        "speedup_vs_baseline": None,
                        "passed": False,
                        "notes": f"timeout: {e}",
                    })
                except Exception as e:
                    tested.append({
                        "config": {"BLOCK_X": bx, "BLOCK_Y": by},
                        "shape": {"H": H, "W": W},
                        "latency_ms": float("inf"),
                        "throughput_gbps": None,
                        "achieved_occupancy": None,
                        "l_inf_error": float("inf"),
                        "ulp_error": None,
                        "baseline_latency_ms": float(baseline_ms),
                        "speedup_vs_baseline": None,
                        "passed": False,
                        "notes": f"exception: {type(e).__name__}: {e}",
                    })
        del inp, out, ref

    passing = [r for r in tested if r.get("passed")]
    best = min(passing, key=lambda r: r["latency_ms"]) if passing else None

    recheck = None
    if best and verify_device is not None and verify_device != device and verify_device < torch.cuda.device_count():
        torch.cuda.synchronize(device)
        torch.cuda.set_device(verify_device)
        apply_vram_quota_gb(vram_gb, verify_device)
        H, W = best["shape"]["H"], best["shape"]["W"]
        bx, by = best["config"]["BLOCK_X"], best["config"]["BLOCK_Y"]
        inp2 = torch.randn((H, W), device=f"cuda:{verify_device}", dtype=torch_dtype)
        out2 = torch.empty_like(inp2)
        ref2 = stencil3x3_reference(inp2.to(torch.float32)).to(torch_dtype)
        def _call2():
            mod.launch(inp2, out2, int(bx), int(by))
        try:
            lat2 = time_ms(_call2, iters=max(10, plan.iters // 2), warmup=max(5, plan.iters // 10), max_seconds=timeout_s)
            linf2 = (out2 - ref2).abs().max().item()
            ulp2 = max_ulp_error(out2, ref2, torch_dtype)
            recheck = {
                "device": int(verify_device),
                "latency_ms": float(lat2),
                "l_inf_error": float(linf2),
                "ulp_error": float(ulp2),
                "passed": bool(linf2 <= plan.tolerance["stencil3x3"][plan.dtype])
            }
        except Exception as e:
            recheck = {"device": int(verify_device), "notes": f"recheck_failed: {e}", "passed": False}
        torch.cuda.set_device(device)

    return Results(
        experiment_id=plan.experiment_id,
        backend="cuda",
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
