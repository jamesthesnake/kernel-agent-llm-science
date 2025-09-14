from __future__ import annotations
import argparse, json, sys
from kernel_agent.grpo.markers import extract_marked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="file with model output text")
    args = ap.parse_args()
    txt = open(args.path).read()
    parsed = extract_marked(txt)
    print(json.dumps({"ok": parsed.ok, "reason": parsed.reason, "think_len": len(parsed.think or ""), "answer_len": len(parsed.answer or "")}, indent=2))
    if not parsed.ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
