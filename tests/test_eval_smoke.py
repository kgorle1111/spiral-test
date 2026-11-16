"""Tiny E1: synthetic tremor injected into 2 real UCI control drawings must be recovered."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spiral import metrics as M

REPO = Path(__file__).resolve().parent.parent
UCI = REPO / "data" / "raw" / "uci"

pytestmark = pytest.mark.skipif(not (UCI / "hw_dataset").exists(), reason="UCI data not downloaded")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("run_eval", REPO / "eval" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._init_worker()
    return mod


def test_injected_tremor_is_recovered_on_real_drawings(ev):
    recs = ev._RECS
    ctrl = [i for i, r in enumerate(recs) if r["group"] == "control" and not r["flags"]][:2]
    assert len(ctrl) == 2
    for i in ctrl:
        center = ev.run_trial({"rec": i, "seed": [0, i], "pp": 0.0})["center"]
        for k, (fk, geom) in enumerate([(5, "linear"), (9, "elliptical"), (11, "radial")]):
            rng = np.random.default_rng([1, i, k])
            p = ev.inj_params(rng, fk, 2.0, geom)
            p["center"] = center
            res = ev.run_trial({"rec": i, "seed": [2, i, k], **p})
            assert res["prom"] >= M.MIN_PROMINENCE and res["amp"] >= M.MIN_RHYTHM_PP_MM, (recs[i]["id"], res)
            assert abs(res["freq"] - p["f"]) <= 0.5, (recs[i]["id"], p["f"], res["freq"])
            assert abs(res["amp"] / res["true_pp"] - 1) <= 0.25, (recs[i]["id"], res["amp"], res["true_pp"])


def test_injection_is_not_a_no_op(ev):
    """Guard against a harness bug where the tremor never reaches the pipeline."""
    i = next(i for i, r in enumerate(ev._RECS) if r["group"] == "control")
    base = ev.run_trial({"rec": i, "seed": [0], "pp": 0.0})
    p = ev.inj_params(np.random.default_rng(3), 7, 2.0, "linear")
    inj = ev.run_trial({"rec": i, "seed": [0], **p})
    assert inj["amp"] > 4 * base["amp"]
