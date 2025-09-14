from __future__ import annotations
import argparse, json, os
from kernel_agent.schemas import TritonPlan, CudaPlan
from kernel_agent.executor import triton_exec, cuda_exec
from kernel_agent.executor.common import device_info, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to plan JSON")
    ap.add_argument("--outdir", default="out", help="Output dir")
    ap.add_argument("--device", type=int, default=0, help="Primary CUDA device index")
    ap.add_argument("--verify_device", type=int, default=None, help="Optional second device index to re-verify best config")
    ap.add_argument("--timeout_s", type=float, default=None, help="Soft wall-time per config (seconds)")
    ap.add_argument("--vram_gb", type=float, default=None, help="Per-process VRAM cap on each used device")
    args = ap.parse_args()

    with open(args.plan) as f:
        raw = json.load(f)

    backend = raw.get("backend")
    exp_id = raw.get("experiment_id", "exp")

    if backend == "triton":
        plan = TritonPlan.model_validate(raw)
        res = triton_exec.run(plan, device=args.device, timeout_s=args.timeout_s,
                              vram_gb=args.vram_gb, verify_device=args.verify_device)
    elif backend == "cuda":
        plan = CudaPlan.model_validate(raw)
        res = cuda_exec.run(plan, device=args.device, timeout_s=args.timeout_s,
                            vram_gb=args.vram_gb, verify_device=args.verify_device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    r = res.model_dump()
    # record executor metadata + quotas across both devices if used
    meta = {
        "primary": device_info(args.device),
        "quotas": {"timeout_s": args.timeout_s, "vram_gb": args.vram_gb},
    }
    if args.verify_device is not None:
        meta["verify"] = device_info(args.verify_device)
    r["executor_info"] = meta

    os.makedirs(args.outdir, exist_ok=True)
    plan_path = os.path.join(args.outdir, f"{exp_id}_plan.json")
    res_path  = os.path.join(args.outdir, f"{exp_id}_results.json")
    write_json(plan_path, raw)
    write_json(res_path, r)
    print(res_path)

if __name__ == "__main__":
    main()
