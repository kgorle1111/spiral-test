"""The web app must run the same Python the evaluation runs, on the data it claims.

Run `uv run python scripts/build_web.py` first; these tests check its output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiral import metrics

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
UCI_KEYS = {"label", "group", "subject", "test", "t_ms", "x_mm", "y_mm", "source", "license"}


@pytest.mark.parametrize("name", ["metrics.py", "trend.py"])
def test_python_copies_are_byte_identical(name):
    src = ROOT / "spiral" / name
    if not src.exists():
        pytest.skip(f"spiral/{name} not written yet")
    dst = WEB / "py" / name
    assert dst.exists(), f"web/py/{name} missing: run scripts/build_web.py"
    assert dst.read_bytes() == src.read_bytes(), f"web/py/{name} drifted: rerun scripts/build_web.py"


def _uci_files():
    return sorted(p for p in (WEB / "data").glob("uci-*.json") if p.name != "uci-index.json")


def test_uci_index_lists_every_export():
    index = json.loads((WEB / "data" / "uci-index.json").read_text())
    files = {p.name for p in _uci_files()}
    assert {e["file"] for e in index} == files
    assert sum(e["group"] == "control" for e in index) >= 2
    assert sum(e["group"] == "parkinson" for e in index) >= 4


def _demo(name):
    return json.loads((WEB / "data" / name).read_text())


@pytest.mark.parametrize("path", _uci_files(), ids=lambda p: p.name)
def test_uci_demo_shape(path):
    d = json.loads(path.read_text())
    assert set(d) == UCI_KEYS
    n = len(d["t_ms"])
    assert n >= 50 and len(d["x_mm"]) == n and len(d["y_mm"]) == n
    assert d["license"] == "CC BY 4.0"
    assert d["group"] in ("control", "parkinson")
    assert all(b >= a for a, b in zip(d["t_ms"], d["t_ms"][1:], strict=False))


def _index():
    return json.loads((WEB / "data" / "uci-index.json").read_text())


@pytest.mark.parametrize("entry", _index(), ids=lambda e: e["file"])
def test_demo_still_produces_its_promised_outcome(entry):
    """Catches metrics.py changes that silently flip a demo (e.g. a gate tightening)."""
    d = _demo(entry["file"])
    r = metrics.analyze(d["t_ms"], d["x_mm"], d["y_mm"])
    expect = entry["expect"]
    if expect == "redo":
        assert r["status"] == "redo" and r["reasons"]
        return
    assert r["status"] == "ok", [x["code"] for x in r["reasons"]]
    want = expect.split(":", 1)[1]
    f = r["metrics"]["freq_hz"]
    if want == "none":
        assert f is None
    elif want != "any":
        lo, hi = (float(v) for v in want.split("-"))
        assert f is not None and lo <= f <= hi


def test_demo_set_shows_both_outcomes():
    kinds = {e["expect"].split(":")[0] for e in _index()}
    assert kinds == {"ok", "redo"}


def test_eval_copy_matches_source_when_present():
    src, dst = ROOT / "results" / "eval.json", WEB / "results" / "eval.json"
    if not src.exists():
        assert not dst.exists(), "web/results/eval.json is stale: results/eval.json is gone"
        return
    assert dst.read_bytes() == src.read_bytes()
