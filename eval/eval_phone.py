"""Evaluate real phone recordings exported from the app ("Download raw drawing").

Put spiral-raw/1 JSON files in data/phone/<device-label>/ and run:
  uv run python eval/eval_phone.py

Per device: effective sample rate, gate pass rate on honest attempts, and
frequency/amplitude recovery of tremor injected into the real phone traces
(4-12 Hz, 1 mm peak-to-peak, random fixed axis). Writes results/phone_eval.json.
With no files it says so and writes nothing: no phone result is ever invented.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spiral import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHONE = ROOT / "data" / "phone"
SEED = 20261116


def load():
    out = []
    for f in sorted(PHONE.glob("*/*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"skip {f}: bad JSON ({e})")
            continue
        if d.get("schema") != "spiral-raw/1":
            print(f"skip {f}: schema {d.get('schema')!r}")
            continue
        d["_device"] = f.parent.name
        d["_file"] = f.name
        out.append(d)
    return out


def inject(d, f_hz, pp_mm, rng):
    t = (np.asarray(d["t_ms"], float) - d["t_ms"][0]) / 1000.0
    ax = rng.uniform(0, math.pi)
    disp = 0.5 * pp_mm * np.sin(2 * math.pi * f_hz * t + rng.uniform(0, 2 * math.pi))
    return (
        np.asarray(d["x_mm"], float) + disp * math.cos(ax),
        np.asarray(d["y_mm"], float) + disp * math.sin(ax),
    )


def main() -> int:
    drawings = load()
    if not drawings:
        print(f"no phone recordings in {PHONE}/<device>/*.json - nothing measured, nothing written")
        return 0
    rng = np.random.default_rng(SEED)
    by_dev: dict[str, dict] = {}
    for d in drawings:
        dev = by_dev.setdefault(d["_device"], {"n": 0, "passed": 0, "rates": [], "reasons": {}, "trials": []})
        dev["n"] += 1
        res = metrics.analyze(
            d["t_ms"], d["x_mm"], d["y_mm"], stroke=d.get("stroke"), template=d.get("template")
        )
        dev["rates"].append(res["quality"]["rate_hz"])
        if res["status"] == "ok":
            dev["passed"] += 1
        for r in res["reasons"]:
            dev["reasons"][r["code"]] = dev["reasons"].get(r["code"], 0) + 1
        if res["status"] != "ok":
            continue
        for f_hz in np.arange(4.0, 12.01, 1.0):
            x, y = inject(d, float(f_hz), 1.0, rng)
            m = metrics.compute(d["t_ms"], x, y, stroke=d.get("stroke"), template=d.get("template"))
            dev["trials"].append(
                {"f": float(f_hz), "f_hat": m["freq_hz"], "prom": m["prominence"], "pp_hat": m["amp_pp_mm"]}
            )
    report = {}
    for name, dev in by_dev.items():
        tr = dev["trials"]
        within = [abs(t["f_hat"] - t["f"]) <= 0.5 and t["prom"] >= metrics.MIN_PROMINENCE for t in tr]
        ratio = [t["pp_hat"] / 1.0 for t in tr]
        report[name] = {
            "drawings": dev["n"],
            "gate_pass_rate": dev["passed"] / dev["n"],
            "median_rate_hz": float(np.median(dev["rates"])),
            "redo_reasons": dev["reasons"],
            "injection_trials": len(tr),
            "freq_within_0_5hz": float(np.mean(within)) if tr else None,
            "amp_ratio_median": float(np.median(ratio)) if tr else None,
            "amp_ratio_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
            if tr
            else None,
        }
        print(name, json.dumps(report[name]))
    out = ROOT / "results" / "phone_eval.json"
    out.write_text(json.dumps({"seed": SEED, "devices": report}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
