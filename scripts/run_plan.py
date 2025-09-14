# kernel-agent-llm-science/scripts/run_plan.py
from __future__ import annotations
import argparse, json, os
from kernel_agent.schemas import TritonPlan, CudaPlan
from kernel_agent.executor import triton_exec, cuda_exec
from kernel_agent.executor.common import device_info, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to plan JSON")
    ap.add_argument("--outdir", default="out", help="Output dir")
    args = ap.parse_args()

    with open(args.plan) as f:
        raw = json.load(f)

    backend = raw.get("backend")
    exp_id = raw.get("experiment_id", "exp")
    if backend == "triton":
        plan = TritonPlan.model_validate(raw)
        res = triton_exec.run(plan)
    elif backend == "cuda":
        plan = CudaPlan.model_validate(raw)
        res = cuda_exec.run(plan)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    r = res.model_dump()
    r["executor_info"] = device_info()
    plan_path = os.path.join(args.outdir, f"{exp_id}_plan.json")
    res_path  = os.path.join(args.outdir, f"{exp_id}_results.json")
    write_json(plan_path, raw)
    write_json(res_path, r)
    print(res_path)

if __name__ == "__main__":
    main()
