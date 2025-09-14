from __future__ import annotations
import argparse, json, glob

def load_results(paths):
    files = []
    for pat in paths:
        files.extend(glob.glob(pat))
    out = []
    for f in sorted(files):
        with open(f) as fh:
            out.append(json.load(fh))
    return out

def collect_records(results):
    recs = []
    for R in results:
        for t in R["tested"]:
            spd = t.get("speedup_vs_baseline")
            if spd is None:
                continue
            recs.append({
                "experiment_id": R["experiment_id"],
                "backend": R["backend"],
                "op": R["op"],
                "config": t["config"],
                "shape": t["shape"],
                "passed": bool(t["passed"]),
                "speedup": float(spd),
            })
    return recs

def fast_p(records, p: float):
    ok = [r for r in records if r["passed"] and r["speedup"] >= p]
    return (len(ok) / len(records)) if records else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", nargs="+", required=True)
    ap.add_argument("--p", type=float, default=1.0)
    args = ap.parse_args()
    res = load_results(args.results_glob)
    recs = collect_records(res)
    f = fast_p(recs, args.p)
    per_exp = {}
    for r in recs:
        per_exp.setdefault(r["experiment_id"], []).append(r)
    detail = {k: fast_p(v, args.p) for k, v in per_exp.items()}
    print(json.dumps({
        "p": args.p,
        "fast_p": f,
        "n": len(recs),
        "per_experiment_fast_p": detail
    }, indent=2))

if __name__ == "__main__":
    main()
