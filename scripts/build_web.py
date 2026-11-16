"""Build the static web app's generated files.

    uv run python scripts/build_web.py

1. Copies spiral/metrics.py and spiral/trend.py byte-for-byte into web/py/, so the
   browser (Pyodide) runs the exact code the evaluation runs.
   tests/test_web_build.py fails if the copies drift.
2. Exports a handful of real UCI 395 drawings (Isenkul et al. 2014, CC BY 4.0) to
   web/data/uci-*.json for the "Try a real recording" screen, and checks that each
   one still produces the outcome its label promises.
3. Copies results/eval.json -> web/results/eval.json when the evaluation has run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spiral import metrics, uci  # noqa: E402

WEB = ROOT / "web"
PY_FILES = ("metrics.py", "trend.py")
LICENSE = "CC BY 4.0"
SOURCE = (
    "UCI Machine Learning Repository, dataset 395: Parkinson Disease Spiral Drawings Using "
    "Digitized Graphics Tablet (Isenkul, Sakar, Kursun 2014). Wacom Cintiq 12WX tablet and pen."
)

# (subject, test, group, expect, label). expect: "ok:<lo>-<hi>" Hz band, "ok:none" = no
# dominant rhythm, or "redo". The build refuses to ship a demo whose outcome changed.
PICKS = [
    ("H_p000-0043", 0, "parkinson", "ok:5.5-6.5", "Parkinson's, large ~6 Hz tremor"),
    ("P_12060002", 1, "parkinson", "ok:5.0-6.5", "Parkinson's, clear ~5.7 Hz tremor"),
    ("H_P000-0035", 1, "parkinson", "ok:6.5-7.5", "Parkinson's, ~7 Hz tremor"),
    ("P_26060002", 0, "parkinson", "ok:none", "Parkinson's, wobble under the 0.45 mm floor"),
    ("H_P000-0011", 1, "parkinson", "redo", "Parkinson's, drawn too fast (rejected)"),
    ("P_11120004", 0, "parkinson", "redo", "Parkinson's, pen lifted mid-spiral (rejected)"),
    ("C_0002", 0, "control", "ok:none", "Control, no dominant rhythm"),
    ("C_0015", 0, "control", "ok:none", "Control, steady hand"),
]
TEST_NAME = {0: "static spiral", 1: "dynamic spiral"}


def copy_python() -> None:
    dst = WEB / "py"
    dst.mkdir(parents=True, exist_ok=True)
    for name in PY_FILES:
        src = ROOT / "spiral" / name
        if not src.exists():
            print(f"warning: {src} missing; web app will run without it")
            continue
        shutil.copyfile(src, dst / name)  # byte-for-byte
        print(f"copied spiral/{name} -> web/py/{name}")


def check(expect: str, res: dict, subject: str) -> None:
    if expect == "redo":
        if res["status"] != "redo":
            raise SystemExit(f"{subject}: expected the gate to reject it, got {res['status']}")
        return
    if res["status"] != "ok":
        raise SystemExit(f"{subject}: expected ok, got redo {[r['code'] for r in res['reasons']]}")
    f = res["metrics"]["freq_hz"]
    want = expect.split(":", 1)[1]
    if want == "none":
        if f is not None:
            raise SystemExit(f"{subject}: expected no dominant rhythm, got {f:.2f} Hz")
    elif want != "any":
        lo, hi = (float(v) for v in want.split("-"))
        if f is None or not lo <= f <= hi:
            raise SystemExit(f"{subject}: expected {lo}-{hi} Hz, got {f}")


def export_uci() -> None:
    dst = WEB / "data"
    dst.mkdir(parents=True, exist_ok=True)
    drawings = {(d.subject, d.test): d for d in uci.load_all()}
    index = []
    for subject, test, group, expect, label in PICKS:
        d = drawings.get((subject, test))
        if d is None:
            raise SystemExit(f"UCI drawing {subject} test {test} not found")
        if d.group != group:
            raise SystemExit(f"{subject}: group {d.group}, expected {group}")
        t0 = float(d.t_ms[0])
        rec = {
            "label": label,
            "group": d.group,
            "subject": d.subject,
            "test": TEST_NAME.get(d.test, str(d.test)),
            "t_ms": [round(float(t) - t0, 3) for t in d.t_ms],
            # 1 micrometre resolution: far below the tablet's own resolution.
            "x_mm": [round(float(v), 3) for v in d.x_mm],
            "y_mm": [round(float(v), 3) for v in d.y_mm],
            "source": SOURCE,
            "license": LICENSE,
        }
        # Check the exported (rounded) data, not the in-memory arrays: this is what ships.
        res = metrics.analyze(rec["t_ms"], rec["x_mm"], rec["y_mm"])
        check(expect, res, subject)
        name = f"uci-{subject.lower()}-t{d.test}.json"
        (dst / name).write_text(json.dumps(rec, separators=(",", ":")))
        f = (res["metrics"] or {}).get("freq_hz")
        summary = f"{res['status']}" + (f" {f:.2f} Hz" if f else "")
        print(f"exported {name}: {len(rec['t_ms'])} points, {summary}")
        index.append(
            {
                "file": name,
                "label": label,
                "group": d.group,
                "subject": d.subject,
                "test": rec["test"],
                "n": len(rec["t_ms"]),
                "expect": expect,
            }
        )
    (dst / "uci-index.json").write_text(json.dumps(index, indent=1))


def copy_eval() -> None:
    src = ROOT / "results" / "eval.json"
    dst = WEB / "results" / "eval.json"
    if src.exists():
        json.loads(src.read_text())  # refuse to ship a broken file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print("copied results/eval.json -> web/results/eval.json")
    else:
        if dst.exists():
            dst.unlink()  # never show a stale result the evaluation no longer backs
        print("results/eval.json not found; Evidence page will say so")


def main() -> int:
    copy_python()
    export_uci()
    copy_eval()
    return 0


if __name__ == "__main__":
    sys.exit(main())
