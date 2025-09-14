# kernel-agent-llm-science/kernel_agent/executor/cuda_exec.py
from __future__ import annotations
from typing import Dict, Any, List
import os, textwrap
import torch
from torch.utils.cpp_extension import load_inline
from .common import time_ms, stencil3x3_reference, elem_size_bytes, dtype_to_torch
from kernel_agent.schemas import CudaPlan, Results, ConfigResult

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
    # Arch for H100 (sm_90)
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
def run(plan: CudaPlan) -> Results:
    assert plan.op == "stencil3x3", "only stencil3x3 supported in CUDA path for now"
    if plan.dtype != "fp32":
        raise NotImplementedError("Round 0 CUDA path implemented for fp32 only. Extend scalar mapping for fp16/bf16.")
    scalar = "float"
    mod = _compile_module(plan.cuda_kernel, scalar, name=f"stencil3x3_{plan.experiment_id}")
    torch_dtype = dtype_to_torch(plan.dtype)
    tested: List[ConfigResult] = []
    for shape in plan.shapes:
        H, W = shape.H, shape.W
        inp = torch.randn((H, W), device="cuda", dtype=torch_dtype)
        ref = stencil3x3_reference(inp.to(torch.float32)).to(torch_dtype)
        out = torch.empty_like(inp)
        bytes_moved = (9 + 1) * H * W * elem_size_bytes(plan.dtype)  # naive 9 reads + 1 write
        for bx in plan.param_grid.get("BLOCK_X", [16]):
            for by in plan.param_grid.get("BLOCK_Y", [16]):
                def _call():
                    mod.launch(inp, out, int(bx), int(by))
                try:
                    lat = time_ms(_call, iters=plan.iters, warmup=max(5, plan.iters // 5))
                    linf = (out - ref).abs().max().item()
                    tol = plan.tolerance["stencil3x3"][plan.dtype]
                    passed = linf <= tol
                    tput = bytes_moved / (lat * 1e-3) / 1e9
                    tested.append({
                        "config": {"BLOCK_X": bx, "BLOCK_Y": by},
                        "shape": {"H": H, "W": W},
                        "latency_ms": float(lat),
                        "throughput_gbps": float(tput),
                        "achieved_occupancy": None,
                        "l_inf_error": float(linf),
                        "passed": bool(passed),
                    })
                except Exception as e:
                    tested.append({
                        "config": {"BLOCK_X": bx, "BLOCK_Y": by},
                        "shape": {"H": H, "W": W},
                        "latency_ms": float("inf"),
                        "throughput_gbps": None,
                        "achieved_occupancy": None,
                        "l_inf_error": float("inf"),
                        "passed": False,
                        "notes": f"exception: {type(e).__name__}: {e}",
                    })
        del inp, out, ref
    passing = [r for r in tested if r.get("passed")]
    best = min(passing, key=lambda r: r["latency_ms"]) if passing else None
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
        executor_info={}
    )
