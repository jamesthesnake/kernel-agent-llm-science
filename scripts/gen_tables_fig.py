# kernel-agent-llm-science/scripts/gen_tables_figs.py
from __future__ import annotations
import os, json, argparse, re
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", required=True, nargs="+", help="e.g., out/*_results.json")
    ap.add_argument("--artifact_dir", default="artifact")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.artifact_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(args.artifact_dir, "generated"), exist_ok=True)

    rows = []
    for path in args.results_glob:
        for p in sorted(glob_safe(path)):
            with open(p) as f:
                r = json.load(f)
            best = r.get("best")
            if best:
                rows.append((r["experiment_id"], r["backend"], r["op"], str(best["shape"]),
                             str(best["config"]), best["latency_ms"], best["throughput_gbps"], best["l_inf_error"]))
                # Figure: latency vs config
                xs = []
                ys = []
                for t in r["tested"]:
                    if t["passed"]:
                        xs.append(str(t["config"]))
                        ys.append(t["latency_ms"])
                if xs:
                    plt.figure()
                    plt.title(f"{r['experiment_id']} latency by config")
                    plt.plot(range(len(xs)), ys, marker="o")
                    plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
                    plt.ylabel("Latency (ms)")
                    plt.tight_layout()
                    figp = os.path.join(args.artifact_dir, "figs", f"{r['experiment_id']}_lat.png")
                    plt.savefig(figp, dpi=200)
                    plt.close()

    # Write LaTeX table (generated file referenced by paper)
    table_path = os.path.join(args.artifact_dir, "generated", "table_main.tex")
    with open(table_path, "w") as f:
        f.write("\\begin{tabular}{l l l l l r r r}\n")
        f.write("\\toprule\n")
        f.write("ID & Backend & Op & Shape & Config & Latency (ms) & GB/s & $\\ell_\\infty$\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" {} & {} & {} & {} & {} & {:.4f} & {:.2f} & {:.2e}\\\\\n".format(*row))
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(table_path)

def glob_safe(pattern):
    import glob
    return glob.glob(pattern)

if __name__ == "__main__":
    main()
