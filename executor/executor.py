#!/usr/bin/env python3
import os, json, time, math, argparse, textwrap, itertools, hashlib, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

# Optional Triton (kept from your earlier flow)
try:
    import triton, triton.language as tl  # noqa: F401
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

# ---------- Utilities ----------
def now_ms(): return int(time.time() * 1000)
def sha256(txt: str) -> str: return hashlib.sha256(txt.encode()).hexdigest()[:12]

@dataclass
class Result:
    cfg: dict
    latency_ms: float
    throughput_gbps: float
    l_inf_error: float
    status: str

# ---------- Reference ops ----------
def row_softmax_ref(x):
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    ex = torch.exp(x - x_max)
    return ex / torch.sum(ex, dim=-1, keepdim=True)

def stencil3x3_ref(img: torch.Tensor, kernel: torch.Tensor):
    # img [H,W], kernel [3,3]; simple valid conv (no padding): output [H,W] with clamped borders
    H, W = img.shape
    out = torch.empty_like(img)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    ii = min(max(i+di, 0), H-1)
                    jj = min(max(j+dj, 0), W-1)
                    acc += img[ii, jj] * kernel[di+1, dj+1]
            out[i,j] = acc
    return out

# ---------- CUDA compile/launch ----------
def compile_cuda_inline(name: str, cuda_src: str):
    from torch.utils.cpp_extension import load_inline
    # H100 targets SM90; keep PTX for forward-compat
    extra_cuda_cflags = [
        "-O3",
        "-gencode", "arch=compute_90,code=sm_90",
        "-gencode", "arch=compute_90,code=compute_90",
        "--use_fast_math",
    ]
    mod = load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,  # expose all global kernels
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    return mod

def launch_stencil3x3_cuda(mod, img: torch.Tensor, k: torch.Tensor,
                           block_x: int, block_y: int) -> torch.Tensor:
    H, W = img.shape
    out = torch.empty_like(img)
    # decide grid/block
    bx, by = block_x, block_y
    gx = (W + bx - 1) // bx
    gy = (H + by - 1) // by
    # Prepare pointers
    func = getattr(mod, "stencil3x3_kernel")
    func(
        img.contiguous(), out, k.contiguous(),
        H, W,
        grid=(gx, gy, 1),
        block=(bx, by, 1),
        stream=torch.cuda.current_stream().cuda_stream
    )
    return out

# ---------- Triton (kept) ----------
def compile_triton(kernel_src: str):
    g = {}
    exec(kernel_src, g)
    ks = [v for k, v in g.items() if callable(v) and k.endswith("_kernel")]
    if not ks:
        raise ValueError("No *_kernel function found in Triton code")
    return ks[0]

def bench_softmax_triton(kernel, dtype, B, N, BLOCK, num_warps, num_stages, iters=50):
    device = torch.device("cuda")
    x = torch.randn((B, N), dtype=dtype, device=device)
    y = torch.empty_like(x)
    ref = row_softmax_ref(x.float()).to(dtype)

    grid = (B,)
    def run_once():
        kernel[grid](x, y, B, N, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)

    for _ in range(5): run_once(); torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters): run_once()
    end.record(); torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / iters

    bytes_per_elem = torch.finfo(dtype).bits // 8
    traffic_bytes = 3 * B * N * bytes_per_elem
    throughput_gbps = (traffic_bytes / (latency_ms / 1000.0)) / 1e9
    l_inf_error = (y - ref).abs().max().item()
    return Result(cfg=dict(B=B,N=N,BLOCK=BLOCK,num_warps=num_warps,num_stages=num_stages),
                  latency_ms=float(latency_ms),
                  throughput_gbps=float(throughput_gbps),
                  l_inf_error=float(l_inf_error),
                  status="ok")

# ---------- Runner ----------
def run_plan(plan: Dict[str, Any], out_dir: Path, visible_device: int):
    torch.cuda.set_device(visible_device)
    exp_id = plan["experiment_id"]
    backend = plan.get("backend", "triton")  # "triton" | "cuda"
    op = plan["op"]
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[plan["dtype"]]
    tolerance = plan["tolerance"][op][plan["dtype"]]
    results: List[Dict[str, Any]] = []

    # Persist original plan
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{exp_id}_plan.json").write_text(json.dumps(plan, indent=2))

    if backend == "triton":
        if not HAVE_TRITON: raise RuntimeError("Triton not available in this image")
        kernel = compile_triton(plan["triton_kernel"])
        for s in plan["shapes"]:
            B, N = int(s["B"]), int(s["N"])
            for BLOCK in plan["param_grid"]["BLOCK"]:
                for W in plan["param_grid"]["num_warps"]:
                    for S in plan["param_grid"]["num_stages"]:
                        r = bench_softmax_triton(kernel, dtype, B, N, int(BLOCK), int(W), int(S))
                        r.status = "ok" if r.l_inf_error <= tolerance else "fail_tolerance"
                        results.append(r.__dict__)
    elif backend == "cuda":
        # Compile once per plan
        cuda_src = plan["cuda_kernel"]
        mod = compile_cuda_inline(f"kb_{exp_id}_{sha256(cuda_src)}", cuda_src)
        for s in plan["shapes"]:
            H, W = int(s["H"]), int(s["W"])
            # inputs on current GPU
            img = torch.randn((H, W), dtype=dtype, device="cuda")
            k = torch.tensor([[0.0625, 0.125, 0.0625],
                              [0.1250, 0.250, 0.1250],
                              [0.0625, 0.125, 0.0625]], dtype=dtype, device="cuda")  # Gaussian-ish
            ref = stencil3x3_ref(img.float(), k.float()).to(dtype)
            for bx in plan["param_grid"]["BLOCK_X"]:
                for by in plan["param_grid"]["BLOCK_Y"]:
                    # Warm-up
                    for _ in range(5): launch_stencil3x3_cuda(mod, img, k, int(bx), int(by))
                    torch.cuda.synchronize()
                    # Timing (KernelBench uses CUDA events + multiple trials) :contentReference[oaicite:2]{index=2}
                    iters = plan.get("iters", 50)
                    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                    start.record()
                    for _ in range(iters):
                        out = launch_stencil3x3_cuda(mod, img, k, int(bx), int(by))
                    end.record(); torch.cuda.synchronize()
                    latency_ms = start.elapsed_time(end) / iters
                    # crude bytes proxy: read 9 + write 1 per pixel
                    bytes_per_elem = torch.finfo(dtype).bits // 8
                    traffic = (10 * H * W * bytes_per_elem)
                    thr = (traffic / (latency_ms / 1000.0)) / 1e9
                    lerr = (out - ref).abs().max().item()
                    status = "ok" if lerr <= tolerance else "fail_tolerance"
                    results.append(dict(cfg=dict(H=H,W=W,BLOCK_X=bx,BLOCK_Y=by),
                                        latency_ms=float(latency_ms),
                                        throughput_gbps=float(thr),
                                        l_inf_error=float(lerr),
                                        status=status))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    ok = [x for x in results if x["status"] == "ok"] or results
    ok.sort(key=lambda x: x["latency_ms"])
    best = ok[0]
    out = dict(
        experiment_id=exp_id, ts_ms=now_ms(), backend=backend, op=op, dtype=plan["dtype"],
        hypothesis=plan.get("hypothesis",""),
        best_cfg=best["cfg"], best_latency_ms=best["latency_ms"],
        best_throughput_gbps=best["throughput_gbps"], results=results
    )
    (out_dir / f"{exp_id}_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

# ---------- Multi-GPU orchestration ----------
def shard_and_run(plan_path: Path, out_dir: Path, devices: List[int]):
    plan = json.loads(plan_path.read_text())
    # Enumerate configurations to shard across devices
    cfgs = []
    if plan.get("backend","triton") == "triton":
            for s in plan["shapes"]:
                for BLOCK in plan["param_grid"]["BLOCK"]:
                    for W in plan["param_grid"]["num_warps"]:
                        for S in plan["param_grid"]["num_stages"]:
                            cfgs.append(dict(type="triton", s=s, BLOCK=BLOCK, W=W, S=S))
    else:
            for s in plan["shapes"]:
                for bx in plan["param_grid"]["BLOCK_X"]:
                    for by in plan["param_grid"]["BLOCK_Y"]:
                        cfgs.append(dict(type="cuda", s=s, BX=bx, BY=by))
    # Simple round-robin by device; run per-device subprocess that filters its portion
    # (for simplicity, we just run whole plan per device with CUDA_VISIBLE_DEVICES=dev and rely on device index)
    import multiprocessing as mp
    def worker(dev):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        run_plan(plan, out_dir, 0)
    with mp.Pool(len(devices)) as pool:
        pool.map(worker, devices)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to plan.json")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--devices", default="0", help="Comma-separated GPU ids, e.g., 0,1,2,3")
    args = ap.parse_args()
    shard_and_run(Path(args.plan), Path(args.out_dir), [int(x) for x in args.devices.split(",")])
