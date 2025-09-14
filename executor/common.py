# kernel-agent-llm-science/kernel_agent/executor/common.py
from __future__ import annotations
import json, time, math, os
from typing import Dict, Any, Optional
import torch

def device_info() -> Dict[str, Any]:
    d = torch.cuda.get_device_properties(0)
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "sm": d.major * 10 + d.minor,
        "name": d.name,
        "multi_processor_count": d.multi_processor_count,
        "total_mem_gb": round(d.total_memory / (1024**3), 2),
        "driver": torch.cuda.driver_version if hasattr(torch.cuda, "driver_version") else None,
    }

def dtype_to_torch(dtype: str):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

def elem_size_bytes(dtype: str) -> int:
    return {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]

@torch.inference_mode()
def time_ms(fn, iters: int = 50, warmup: int = 10) -> float:
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters

def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def softmax_reference(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Compute in fp32 for stability; cast to request dtype at end.
    ref = torch.nn.functional.softmax(x.to(torch.float32), dim=1)
    return ref.to(dtype)

def stencil3x3_reference(inp: torch.Tensor) -> torch.Tensor:
    # clamped border (replicate padding), channel-less 2D.
    # inp: [H, W]
    H, W = inp.shape
    out = torch.empty_like(inp)
    for y in range(H):
        y0 = max(y-1, 0); y1 = y; y2 = min(y+1, H-1)
        for x in range(W):
            x0 = max(x-1, 0); x1 = x; x2 = min(x+1, W-1)
            s = (inp[y0, x0] + inp[y0, x1] + inp[y0, x2] +
                 inp[y1, x0] + inp[y1, x1] + inp[y1, x2] +
                 inp[y2, x0] + inp[y2, x1] + inp[y2, x2]) / 9.0
            out[y, x] = s
    return out
