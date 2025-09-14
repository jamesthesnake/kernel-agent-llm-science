from __future__ import annotations
import random

def rewrite_spec(op: str, shape: dict, dtype: str, backend: str) -> tuple[str, dict]:
    """Simple problem-rewriting: shuffle constraint order and add a harmless context line."""
    ctx = [
        f"Target GPU SKU: H100 (SM90).",
        f"Toolchain: CUDA 12.4; PyTorch 2.4; Triton 3.0.",
        f"Sandbox: no network; time/VRAM quotas apply."
    ]
    random.shuffle(ctx)
    prompt_lines = [
        "SYSTEM+USER PRIMER — KernelLLM-Agent (Markers enforced)",
        "",
        f"Task: backend={backend}, op={op}, dtype={dtype}, shape={shape}.",
        "You MUST respond using the boundary tokens and output ONE JSON PLAN in the answer block.",
        "No prose outside the markers.",
        "",
        *ctx,
        "",
        "PLAN schema reminder (one backend only):",
        "- Triton row_softmax plan with fields: experiment_id, backend, op, dtype, shapes, hypothesis, metrics, tolerance, param_grid, triton_kernel.",
        "- CUDA stencil plan with fields: experiment_id, backend, op, dtype, shapes, hypothesis, metrics, tolerance, param_grid, iters, cuda_kernel.",
    ]
    meta = {"ctx_variant": ctx}
    return "\n".join(prompt_lines), meta
