# verify_best.py
import json, sys, argparse
from pathlib import Path
import subprocess

def run(plan_path: Path, out_dir: Path, device: int):
    cmd = [
        "bash","-lc",
        f". .venv/bin/activate && python executor.py --plan {plan_path} --out_dir {out_dir} --devices {device}"
    ]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--orig_plan", required=True)
    ap.add_argument("--verify_device", type=int, default=0)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    plan = json.loads(Path(args.orig_plan).read_text())
    backend = plan.get("backend","triton")
    op = plan["op"]; dtype = plan["dtype"]; shapes = plan["shapes"]

    best = res["best_cfg"]
    verify = {
        "experiment_id": res["experiment_id"] + "_verify",
        "backend": backend,
        "op": op,
        "dtype": dtype,
        "shapes": shapes,
        "hypothesis": "Verification run of best config on a held-out GPU.",
        "tolerance": plan["tolerance"],
    }

    if backend == "triton":
        verify["triton_kernel"] = plan["triton_kernel"]
        verify["param_grid"] = {
            "BLOCK": [best["BLOCK"]],
            "num_warps": [best["num_warps"]],
            "num_stages": [best["num_stages"]],
        }
    else:
        verify["cuda_kernel"] = plan["cuda_kernel"]
        verify["param_grid"] = {
            "BLOCK_X": [best["BLOCK_X"]],
            "BLOCK_Y": [best["BLOCK_Y"]],
        }
        verify["iters"] = 200

    vpath = Path("plans") / "plan_verify.json"
    vpath.parent.mkdir(parents=True, exist_ok=True)
    vpath.write_text(json.dumps(verify, indent=2))
    run(vpath, Path(args.out_dir), args.verify_device)
