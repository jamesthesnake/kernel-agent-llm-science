from __future__ import annotations
import argparse
from grpo.grpo_loop import train_grpo, Task
from providers.transformers_local import HFPolicy, HFFrozenRef

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or path (policy)")
    ap.add_argument("--ref", required=True, help="HF model name or path (frozen reference)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--verify_device", type=int, default=None)
    ap.add_argument("--timeout_s", type=float, default=2.0)
    ap.add_argument("--vram_gb", type=float, default=16.0)
    ap.add_argument("--groups_per_step", type=int, default=2)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--outdir", default="grpo_logs")
    args = ap.parse_args()

    # Minimal task set (extend as needed)
    tasks = [
        Task(backend="triton", op="row_softmax", dtype="bf16", shape={"B": 32, "N": 8192}),
        Task(backend="cuda",   op="stencil3x3", dtype="fp32", shape={"H": 2048, "W": 2048}),
    ]

    policy = HFPolicy(args.model, lr=5e-6, device="cuda", policy_gpus=(0,1))
    ref    = HFFrozenRef(args.ref, device=f"cuda:{args.device}")
    train_grpo(
        policy, ref, tasks,
        device_index=args.device,
        verify_device=args.verify_device,
        timeout_s=args.timeout_s,
        vram_gb=args.vram_gb,
        groups_per_step=args.groups_per_step,
        G=args.G,
        outdir=args.outdir
    )

if __name__ == "__main__":
    main()
