#!/usr/bin/env python3
"""
agent/agent.py — Orchestrates PLAN → (optional) EXECUTE → (optional) ANALYZE → next PLAN.

Authoring lane only: runs outside the sandboxed executor container.
- Uses an OpenAI-compatible /v1/chat/completions endpoint (vLLM or a provider).
- Emits PLAN JSON files under ../plans/.
- (Optional) Executes your executor (local or via a shell command you provide).
- (Optional) Feeds results back to the LLM to get the next PLAN.

ENV (set these to point at DeepSeek-R1-Distill-Llama-70B via vLLM or a provider):
  OPENAI_BASE_URL   (e.g., http://localhost:8000/v1)
  OPENAI_API_KEY    (any string for vLLM; provider key if remote)
  OPENAI_MODEL      (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

Typical usage:
  # 1) Generate a single PLAN (triton softmax):
  python agent.py --rounds 1 --backend triton --op row_softmax --dtype bf16 --B 32 --N 8192

  # 2) Generate a single PLAN (cuda stencil):
  python agent.py --rounds 1 --backend cuda --op stencil3x3 --dtype fp32 --H 2048 --W 2048

  # 3) Do PLAN → EXECUTE (locally) → ANALYZE → next PLAN (no docker; requires local venv):
  python agent.py --rounds 2 --backend triton --op row_softmax --dtype bf16 --B 32 --N 8192 \
                  --exec_cmd "python executor/executor.py --plan {plan} --out_dir out --devices 0"

  # 4) Do PLAN → EXECUTE in docker (you supply the docker run command template):
  python agent.py --rounds 1 --backend cuda --op stencil3x3 --dtype fp32 --H 2048 --W 2048 \
                  --exec_cmd "docker run --rm --gpus '\"device=1\"' --network=none -v $PWD:/ws kernelllm-agent:local \
                               bash -lc '. .venv/bin/activate && python executor/executor.py --plan /ws/{plan} --out_dir /ws/out --devices 0'"

Notes:
- Keep the executor air-gapped; this script only *calls* it.
- The LLM must return ONE JSON object; we hard-extract the first {...} block defensively.
"""

from __future__ import annotations
import os, json, subprocess, sys, argparse, textwrap
from pathlib import Path
from datetime import datetime

# ---------- Paths (repo-root aware) ----------
AGENT_DIR = Path(__file__).resolve().parent
ROOT = AGENT_DIR.parent
PLANS_DIR = ROOT / "plans"
OUT_DIR = ROOT / "out"

# ---------- Prompts ----------
PLAN_SCHEMA_HINT = r"""
You are KernelLLM-Agent. Output ONE JSON object (no prose, no code fences) matching the schema:

For Triton:
{
  "experiment_id": "softmax_bf16_round<N>",
  "backend": "triton",
  "op": "row_softmax",
  "dtype": "bf16",                       // fp32 | bf16 | fp16
  "shapes": [ { "B": 32, "N": 8192 } ],
  "hypothesis": "Short falsifiable statement about BLOCK/warps/stages vs. metrics.",
  "metrics": ["latency_ms","throughput_gbps","achieved_occupancy","l_inf_error"],
  "tolerance": { "row_softmax": { "bf16": 1e-2, "fp32": 1e-5 } },
  "param_grid": { "BLOCK": [64,128,256], "num_warps":[2,4,8], "num_stages":[1,2] },
  "triton_kernel": "<STRING WITH FULL @triton.jit KERNEL ending with _kernel>"
}

For CUDA:
{
  "experiment_id": "stencil_fp32_round<N>",
  "backend": "cuda",
  "op": "stencil3x3",
  "dtype": "fp32",                       // fp32 | bf16 | fp16
  "shapes": [ { "H": 2048, "W": 2048 } ],
  "hypothesis": "Short falsifiable statement about BLOCK_X/BLOCK_Y vs. metrics.",
  "metrics": ["latency_ms","throughput_gbps","l_inf_error"],
  "tolerance": { "stencil3x3": { "fp32": 1e-6, "bf16": 5e-3, "fp16": 5e-3 } },
  "param_grid": { "BLOCK_X": [16,32], "BLOCK_Y": [8,16] },
  "iters": 50,
  "cuda_kernel": "<STRING WITH FULL __global__ stencil3x3_kernel(...) implementation>"
}

Rules:
- Return a single raw JSON object ONLY (no commentary).
- Kernel code must compile; no filesystem or network calls inside the kernel.
- Use numerically stable math where relevant (subtract-max for softmax).
- Keep JSON <= 50KB.
"""

ANALYZE_HINT = r"""
You are KernelLLM-Agent. I will give you the executor results JSON (from the previous PLAN).
Tasks:
1) Decide if your hypothesis was supported, partially supported, or contradicted (use numbers).
2) Diagnose perf bottleneck (compute vs memory; register spills; occupancy if measured).
3) Emit the NEXT PLAN JSON (same schema as before) with improved kernel or tuned param_grid.
Output ONE raw JSON object only (no prose).
"""

# ---------- LLM call (OpenAI-compatible /chat/completions) ----------
def llm_chat_once(prompt: str) -> str:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are KernelLLM-Agent. Return ONE JSON object only (no prose)."},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 6000
    }
    pj = json.dumps(payload).replace("'", "'\"'\"'")
    cmd = [
        "bash","-lc",
        f"curl -s {base}/chat/completions "
        "-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {api_key}' "
        f"-d '{pj}'"
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    out = data["choices"][0]["message"]["content"]
    # Extract first {...} block defensively
    s, e = out.find("{"), out.rfind("}")
    if s < 0 or e < 0:
        raise ValueError("LLM did not return a JSON object")
    return out[s:e+1]

# ---------- Baseline seed plans (if LLM unavailable) ----------
def seed_triton_softmax(dtype: str, B: int, N: int) -> str:
    kernel = (
        "@triton.jit\n"
        "def row_softmax_kernel(X_ptr, Y_ptr, B, N, BLOCK: tl.constexpr):\n"
        "    row_id = tl.program_id(0)\n"
        "    offs = row_id * N + tl.arange(0, BLOCK)\n"
        "    mask = tl.arange(0, BLOCK) < N\n"
        "    x_in = tl.load(X_ptr + offs, mask=mask, other=-float('inf'))\n"
        "    x = x_in.to(tl.float32)\n"
        "    x_max = tl.max(x, axis=0)\n"
        "    x = x - x_max\n"
        "    ex = tl.exp(x)\n"
        "    denom = tl.sum(ex, axis=0)\n"
        "    y = (ex / denom).to(x_in.dtype)\n"
        "    tl.store(Y_ptr + offs, y, mask=mask)\n"
    )
    plan = {
        "experiment_id": f"softmax_{dtype}_round0",
        "backend": "triton",
        "op": "row_softmax",
        "dtype": dtype,
        "shapes": [ {"B": int(B), "N": int(N)} ],
        "hypothesis": "BLOCK≈N with 4 warps minimizes inactive lanes without heavy spills.",
        "metrics": ["latency_ms","throughput_gbps","achieved_occupancy","l_inf_error"],
        "tolerance": { "row_softmax": { "bf16": 1e-2, "fp32": 1e-5 } },
        "param_grid": { "BLOCK": [N, 2*N], "num_warps":[2,4,8], "num_stages":[1,2] },
        "triton_kernel": kernel
    }
    return json.dumps(plan, indent=2)

def seed_cuda_stencil(dtype: str, H: int, W: int) -> str:
    cuda_src = r'''
#include <cuda.h>
#include <cuda_runtime.h>
extern "C" __global__ void stencil3x3_kernel(
    const float* __restrict__ img, float* __restrict__ out, const float* __restrict__ k,
    int H, int W) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  float acc = 0.f;
  #pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
    int yy = y + dy; yy = min(max(yy, 0), H-1);
    #pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      int xx = x + dx; xx = min(max(xx, 0), W-1);
      float kv = k[(dy+1)*3 + (dx+1)];
      acc += img[yy*W + xx] * kv;
    }
  }
  out[y*W + x] = acc;
}
'''.strip("\n")
    plan = {
        "experiment_id": f"stencil_{dtype}_round0",
        "backend": "cuda",
        "op": "stencil3x3",
        "dtype": dtype,
        "shapes": [ {"H": int(H), "W": int(W)} ],
        "hypothesis": "16x16 and 32x8 balance occupancy vs. memory traffic for 3x3 stencil.",
        "metrics": ["latency_ms","throughput_gbps","l_inf_error"],
        "tolerance": { "stencil3x3": { "fp32": 1e-6, "bf16": 5e-3, "fp16": 5e-3 } },
        "param_grid": { "BLOCK_X": [16, 32], "BLOCK_Y": [8, 16] },
        "iters": 50,
        "cuda_kernel": cuda_src
    }
    return json.dumps(plan, indent=2)

# ---------- Helpers ----------
def write_text(path: Path, txt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)

def run_exec(exec_cmd_tpl: str, plan_path: Path):
    """
    Execute the plan by invoking your executor.
    exec_cmd_tpl: a format string with {plan} placeholder, executed via bash -lc
    Example: "python executor/executor.py --plan {plan} --out_dir out --devices 0"
    """
    cmd = exec_cmd_tpl.format(plan=str(plan_path).replace("\\", "\\\\"))
    print(f"[EXEC] {cmd}")
    subprocess.check_call(["bash","-lc", cmd])

def find_latest_results(out_dir: Path, exp_id_prefix: str) -> Path | None:
    cands = sorted(out_dir.glob(f"{exp_id_prefix}*results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1, help="Number of agent rounds (PLAN + optional ANALYZE)")
    ap.add_argument("--backend", choices=["triton","cuda"], required=True)
    ap.add_argument("--op", type=str, default="row_softmax")
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--B", type=int, default=32, help="For row_softmax")
    ap.add_argument("--N", type=int, default=8192, help="For row_softmax")
    ap.add_argument("--H", type=int, default=2048, help="For stencil3x3")
    ap.add_argument("--W", type=int, default=2048, help="For stencil3x3")
    ap.add_argument("--exec_cmd", type=str, default="", help="Shell command template to run executor; include {plan}")
    args = ap.parse_args()

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Build the prompting context for ROUND 0 ----
    if args.backend == "triton":
        user_prompt = (f"ROUND 0 START: {args.op} (B={args.B}, N={args.N}, dtype={args.dtype})\n\n" + PLAN_SCHEMA_HINT)
    else:
        user_prompt = (f"ROUND 0 START: {args.op} (H={args.H}, W={args.W}, dtype={args.dtype})\n\n" + PLAN_SCHEMA_HINT)

    # Try LLM; if unavailable, seed a baseline plan
    try:
        plan0_json = llm_chat_once(user_prompt)
    except Exception as e:
        print(f"[WARN] LLM failed for round 0 ({e}); using seed plan.")
        plan0_json = seed_triton_softmax(args.dtype, args.B, args.N) if args.backend=="triton" \
                     else seed_cuda_stencil(args.dtype, args.H, args.W)

    plan0 = json.loads(plan0_json)
    if "backend" not in plan0:
        plan0["backend"] = args.backend  # normalize if model forgot
        plan0_json = json.dumps(plan0, indent=2)

    plan_path = PLANS_DIR / f"plan_round0.json"
    write_text(plan_path, plan0_json)
    print(f"[PLAN0] {plan_path.relative_to(ROOT)}")

    # ---- Optional EXECUTE ----
    if args.exec_cmd:
        run_exec(args.exec_cmd, plan_path)
        exp_id = plan0.get("experiment_id","")
        rpath = find_latest_results(OUT_DIR, exp_id) or find_latest_results(OUT_DIR, exp_id.split("_round")[0])
        if not rpath:
            print("[INFO] No results found after execution. Stopping.")
            return
        print(f"[RESULT0] {rpath.relative_to(ROOT)}")

    # ---- Optional ANALYZE → Next PLAN(s) ----
    rounds = max(1, args.rounds)
    prev_results_path = rpath if args.exec_cmd else None
    for i in range(1, rounds):
        if prev_results_path is None:
            print("[INFO] Skipping ANALYZE (no results).")
            break
        try:
            results_json = json.loads(prev_results_path.read_text())
            analyze_prompt = ANALYZE_HINT + "\n\n" + json.dumps(results_json)
            next_plan_json = llm_chat_once(analyze_prompt)
        except Exception as e:
            print(f"[WARN] LLM analyze failed ({e}); stopping.")
            break
        next_plan = json.loads(next_plan_json)
        if "backend" not in next_plan:
            next_plan["backend"] = args.backend
            next_plan_json = json.dumps(next_plan, indent=2)
        npath = PLANS_DIR / f"plan_round{i}.json"
        write_text(npath, next_plan_json)
        print(f"[PLAN{i}] {npath.relative_to(ROOT)}")
        if args.exec_cmd:
            run_exec(args.exec_cmd, npath)
            exp_id = next_plan.get("experiment_id","")
            prev_results_path = find_latest_results(OUT_DIR, exp_id) or find_latest_results(OUT_DIR, exp_id.split("_round")[0])
            if prev_results_path:
                print(f"[RESULT{i}] {prev_results_path.relative_to(ROOT)}")
            else:
                print("[INFO] No results after execution; stopping.")
                break

    print("[DONE] Plans in ./plans ; results (if executed) in ./out")

if __name__ == "__main__":
    main()
