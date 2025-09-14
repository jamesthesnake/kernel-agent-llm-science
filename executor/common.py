from __future__ import annotations
import json, time, math, os
from typing import Dict, Any, Optional
import torch

def device_info(dev: int = 0) -> Dict[str, Any]:
    d = torch.cuda.get_device_properties(dev)
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "sm": d.major * 10 + d.minor,
        "name": d.name,
        "device_index": dev,
        "multi_processor_count": d.multi_processor_count,
        "total_mem_gb": round(d.total_memory / (1024**3), 2),
        "driver": torch.cuda.driver_version if hasattr(torch.cuda, "driver_version") else None,
    }

def dtype_to_torch(dtype: str):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

def elem_size_bytes(dtype: str) -> int:
    return {"fp32": 4, "bf16": 2, "fp16": 2}[dtype]

@torch.inference_mode()
def time_ms(fn, iters: int = 50, warmup: int = 10, max_seconds: Optional[float] = None) -> float:
    """CUDA-event timing with a soft wall-time budget. If the loop would exceed
    max_seconds, early-abort and raise TimeoutError."""
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt.record()
    t0 = time.perf_counter()
    done = 0
    for i in range(iters):
        fn()
        done += 1
        if max_seconds is not None and (time.perf_counter() - t0) > max_seconds:
            end_evt.record()
            torch.cuda.synchronize()
            # Use what we have to avoid poisoning the run with inf
            ms = start_evt.elapsed_time(end_evt) / max(1, done)
            raise TimeoutError(f"timed out after {max_seconds}s (partial mean {ms:.4f} ms over {done} iters)")
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / iters

@torch.inference_mode()
def max_ulp_error(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> float:
    """
    Compute max ULP error w.r.t. the *target dtype*’s representable grid.
    We quantize both to `dtype`, compute nextafter spacing in that dtype,
    then measure |out-ref| / ulp_spacing.
    """
    dev = out.device
    out_d = out.to(dtype)
    ref_d = ref.to(dtype)

    # one-step toward +inf in the target dtype
    pos_inf = torch.tensor(float("inf"), dtype=dtype, device=dev)
    step = (torch.nextafter(ref_d, pos_inf) - ref_d).to(torch.float32).abs()

    # Guard: zero step can only happen at inf/NaN; not expected in softmax, but be safe.
    finfo = torch.finfo(dtype)
    tiny = torch.tensor(finfo.tiny, dtype=torch.float32, device=dev)
    step = torch.where(step == 0, tiny, step)

    ulp = (out_d.to(torch.float32) - ref_d.to(torch.float32)).abs() / step
    return float(torch.nan_to_num(ulp, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item())

def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def softmax_reference(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    ref = torch.nn.functional.softmax(x.to(torch.float32), dim=1)
    return ref.to(dtype)

def stencil3x3_reference(inp: torch.Tensor) -> torch.Tensor:
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

def apply_vram_quota_gb(vram_gb: Optional[float], device: int):
    if vram_gb is None: 
        return
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    frac = max(0.05, min(0.95, vram_gb / total))
    torch.cuda.set_per_process_memory_fraction(frac, device=device)
