"""Command line: analyze one drawing, or draft a note from exported history.

uv run spiral analyze path/to/spiral-raw.json      # a drawing exported from the app
uv run spiral analyze data/raw/uci/hw_dataset/parkinson/P_02100001.txt --test 0
uv run spiral note path/to/history.json [--hand right]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spiral import metrics


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def cmd_analyze(args) -> int:
    p = Path(args.path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if d.get("schema") != "spiral-raw/1":
            print(f"{p}: not a spiral-raw/1 file (schema={d.get('schema')!r})", file=sys.stderr)
            return 2
        res = metrics.analyze(
            d["t_ms"], d["x_mm"], d["y_mm"], stroke=d.get("stroke"), template=d.get("template")
        )
    else:
        from spiral import uci

        arr = uci._read(p)
        sel = arr[arr[:, 6] == args.test]
        if sel.shape[0] < 10:
            print(f"{p}: no samples for test {args.test}", file=sys.stderr)
            return 2
        res = metrics.analyze(sel[:, 5], sel[:, 0] * uci.MM_PER_PX, -sel[:, 1] * uci.MM_PER_PX)
    print(f"status: {res['status']}")
    for r in res["reasons"]:
        print(f"  redo: {r['message']}")
    print("quality: " + ", ".join(f"{k}={_fmt(v)}" for k, v in res["quality"].items()))
    if res["metrics"]:
        print("metrics: " + ", ".join(f"{k}={_fmt(v)}" for k, v in res["metrics"].items()))
    return 0 if res["status"] == "ok" else 1


def cmd_note(args) -> int:
    from spiral import note, trend

    records = json.loads(Path(args.path).read_text())
    if isinstance(records, dict):
        records = records.get("records", [])
    summary = trend.summarize(records)
    out = note.draft_note(summary, hand=args.hand)
    print(out["text"])
    print(f"\n[source: {out['source']}; breaches caught: {len(out.get('breaches', []))}]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spiral")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze one drawing")
    a.add_argument("path")
    a.add_argument("--test", type=int, default=0, help="UCI test id (0 static, 1 dynamic)")
    a.set_defaults(fn=cmd_analyze)
    n = sub.add_parser("note", help="draft the neurologist note from exported history")
    n.add_argument("path")
    n.add_argument("--hand", default="right")
    n.set_defaults(fn=cmd_note)
    args = ap.parse_args(argv)
    np.seterr(all="ignore")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
