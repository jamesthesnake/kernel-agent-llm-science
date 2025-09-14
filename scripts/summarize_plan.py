# kernel-agent-llm-science/scripts/summarize_results.py
from __future__ import annotations
import json, argparse
from rich.console import Console
from rich.table import Table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    with open(args.results) as f:
        r = json.load(f)
    c = Console()
    c.print(f"[bold]{r['experiment_id']}[/] backend={r['backend']} op={r['op']} dtype={r['dtype']}")
    t = Table("shape", "config", "latency(ms)", "GB/s", "Linf", "pass")
    for item in r["tested"]:
        shape = str(item["shape"])
        conf  = str(item["config"])
        t.add_row(shape, conf,
                  f"{item['latency_ms']:.4f}" if item["latency_ms"] != float("inf") else "inf",
                  f"{item['throughput_gbps']:.2f}" if item["throughput_gbps"] else "-",
                  f"{item['l_inf_error']:.3e}",
                  "✓" if item["passed"] else "×")
    c.print(t)
    if r.get("best"):
        c.print("[bold green]BEST[/]:", r["best"])

if __name__ == "__main__":
    main()
