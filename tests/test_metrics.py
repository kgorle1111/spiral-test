"""Unit tests for spiral/metrics.py on synthetic spirals.

Synthetic data only proves the maths; the evaluation on real drawings is eval/run_eval.py.
"""

import math

import numpy as np
import pytest

from spiral import metrics

# Phone-sized template: outer radius a + b * 6*pi = 30 mm; traced in 15 s that is ~21 mm/s.
TPL = {"cx": 0.0, "cy": 0.0, "a": 4.0, "b": 26.0 / (6 * math.pi), "turns": 3.0, "phi0": 0.0, "dir": 1}


def spiral(duration_s=15.0, fs=60.0, tremor_hz=None, pp_mm=0.0, axis_deg=30.0, jitter_ms=0.0, seed=0):
    """A traced 3-turn template spiral with optional linear tremor along a fixed axis."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    if jitter_ms:
        t = t + rng.uniform(-jitter_ms, jitter_ms, n) / 1000.0
    th = TPL["turns"] * 2 * math.pi * t / duration_s
    r = TPL["a"] + TPL["b"] * th
    x, y = r * np.cos(th), r * np.sin(th)
    if tremor_hz:
        d = 0.5 * pp_mm * np.sin(2 * math.pi * tremor_hz * t + 0.7)
        ax = math.radians(axis_deg)
        x, y = x + d * math.cos(ax), y + d * math.sin(ax)
    return t * 1000.0, x, y


@pytest.mark.parametrize("f", [4.0, 6.0, 8.5, 11.0])
def test_recovers_frequency_and_amplitude(f):
    t, x, y = spiral(tremor_hz=f, pp_mm=1.0)
    res = metrics.analyze(t, x, y, template=TPL)
    assert res["status"] == "ok", res["reasons"]
    m = res["metrics"]
    assert m["dominant_rhythm"]
    assert abs(m["freq_hz"] - f) < 0.25
    assert abs(m["amp_pp_mm"] - 1.0) < 0.25


def test_amplitude_independent_of_axis():
    amps = [
        metrics.compute(*spiral(tremor_hz=6.0, pp_mm=2.0, axis_deg=a))["amp_pp_mm"] for a in (0, 45, 90, 135)
    ]
    assert max(amps) - min(amps) < 0.2


def test_clean_spiral_has_no_dominant_rhythm():
    t, x, y = spiral()
    res = metrics.analyze(t, x, y, template=TPL)
    assert res["status"] == "ok"
    assert res["metrics"]["freq_hz"] is None
    assert res["metrics"]["amp_pp_mm"] < 0.1


def test_jittered_timestamps_still_recover():
    t, x, y = spiral(tremor_hz=7.0, pp_mm=1.0, jitter_ms=2.0)
    m = metrics.compute(t, x, y, template=TPL)
    assert abs(m["freq_hz"] - 7.0) < 0.3


def test_gate_rejects_low_rate():
    t, x, y = spiral(fs=15.0, tremor_hz=6.0, pp_mm=1.0)
    res = metrics.analyze(t, x, y, template=TPL)
    assert res["status"] == "redo"
    assert "low_rate" in [r["code"] for r in res["reasons"]]
    assert res["metrics"] is None


def test_gate_rejects_short_fast_drawing():
    t, x, y = spiral(duration_s=3.0)
    codes = [r["code"] for r in metrics.analyze(t, x, y, template=TPL)["reasons"]]
    assert "too_short" in codes
    assert "too_fast" in codes


def test_gate_rejects_partial_spiral():
    t, x, y = spiral()
    half = t.size // 2
    codes = [r["code"] for r in metrics.analyze(t[:half], x[:half], y[:half], template=TPL)["reasons"]]
    assert "coverage" in codes


def test_gate_rejects_lifts_and_gaps():
    t, x, y = spiral()
    stroke = np.zeros_like(t)
    stroke[t.size // 3 :] = 1
    stroke[2 * t.size // 3 :] = 2
    codes = [r["code"] for r in metrics.analyze(t, x, y, stroke=stroke, template=TPL)["reasons"]]
    assert "lifts" in codes
    keep = np.r_[0 : t.size // 2, t.size // 2 + 30 : t.size]  # drop 0.5 s of samples
    codes = [r["code"] for r in metrics.analyze(t[keep], x[keep], y[keep], template=TPL)["reasons"]]
    assert "gap" in codes


def test_gate_rejects_scribble_off_template():
    t, x, y = spiral()
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.8, t.size))
    y = np.cumsum(rng.normal(0, 0.8, t.size))
    codes = [r["code"] for r in metrics.analyze(t, x, y, template=TPL)["reasons"]]
    assert "off_template" in codes or "coverage" in codes


def test_template_deviation_zero_on_template_and_direction():
    t, x, y = spiral()
    assert metrics.template_deviation(x, y, TPL) < 0.05
    # Same points, template declared the other way round: should not match.
    assert metrics.template_deviation(x, y, {**TPL, "dir": -1}) > 2.0


def test_duplicate_timestamps_and_nans_are_cleaned():
    t, x, y = spiral(tremor_hz=6.0, pp_mm=1.0)
    t2 = np.repeat(t, 2)
    x2, y2 = np.repeat(x, 2), np.repeat(y, 2)
    x2[5] = np.nan
    m = metrics.compute(t2, x2, y2, template=TPL)
    assert abs(m["sampling"]["rate_hz"] - 60.0) < 2.0
    assert abs(m["freq_hz"] - 6.0) < 0.25


def test_fit_center_finds_offset_spiral():
    t, x, y = spiral()
    cx, cy = metrics.fit_center(x + 12.0, y - 7.0)
    assert abs(cx - 12.0) < 1.0 and abs(cy + 7.0) < 1.0


def test_rejects_tiny_input():
    with pytest.raises(ValueError):
        metrics.compute([0, 1, 2], [0, 1, 2], [0, 1, 2])


def test_series_for_plot_is_json_safe():
    import json

    t, x, y = spiral(tremor_hz=6.0, pp_mm=1.0)
    res = metrics.analyze(t, x, y, template=TPL)
    s = metrics.series_for_plot(res["series"])
    json.dumps(s)
    assert len(s["t"]) <= 1200


def test_small_tremor_below_amplitude_floor_is_not_a_rhythm():
    t, x, y = spiral(tremor_hz=6.0, pp_mm=0.5 * metrics.MIN_RHYTHM_PP_MM)
    res = metrics.analyze(t, x, y, template=TPL)
    assert res["status"] == "ok"
    assert res["metrics"]["prominence"] >= metrics.MIN_PROMINENCE  # the peak is there...
    assert not res["metrics"]["dominant_rhythm"]  # ...but too small to call a rhythm
    assert res["metrics"]["freq_hz"] is None


def test_spectral_peak_interpolation_only_at_a_vertex_and_clamped():
    f = np.arange(201) * 0.25
    # Rising through the band edge: the in-band max (12 Hz) is not a vertex. Old code returned 0.0 Hz.
    fp, prom = metrics.spectral_peak(f, f**2)
    assert fp == 12.0 and prom == 0.0
    # A lopsided but genuine peak: the offset never leaves +-half a bin.
    psd = np.ones_like(f)
    psd[23:26] = [5.0, 5.2, 0.1]  # bins 5.75, 6.0, 6.25 Hz
    fp, _ = metrics.spectral_peak(f, psd)
    assert 5.875 <= fp <= 6.0


def test_clock_jump_abstains_instead_of_inventing_a_rhythm():
    n = 3000
    t = np.arange(n) * 9.0
    th = np.linspace(0, 6 * np.pi, n)
    x, y = (5 + 3 * th) * np.cos(th), (5 + 3 * th) * np.sin(th)
    t[1500:] -= 1300.0  # the clock jumps back 1.3 s mid-recording
    m = metrics.compute(t, x, y)
    assert m["sampling"]["clock_backsteps"] > 100
    assert m["prominence"] < metrics.MIN_PROMINENCE  # sorting used to give an 11 Hz peak, prominence ~440
    res = metrics.analyze(t, x, y)
    assert res["status"] == "redo"
    assert "clock" in [r["code"] for r in res["reasons"]]
    # Equal timestamps are not a clock jump.
    assert metrics.clock_backsteps(np.repeat(np.arange(10.0), 2)) == 0


def test_speed_ignores_the_bridge_across_a_lift():
    t, x, y = spiral()
    n = t.size
    stroke = (np.arange(n) >= n // 2).astype(int)
    keep = np.r_[0 : n // 2, n // 2 + 30 : n]  # pen up for 0.5 s
    t, x, y, stroke = t[keep], x[keep].copy(), y[keep].copy(), stroke[keep]
    still = metrics.compute(t, x, y, stroke)["speed_mm_s"]
    x[stroke == 1] += 40.0  # the pen moved 40 mm while lifted
    moved = metrics.compute(t, x, y, stroke)["speed_mm_s"]
    # Counting the 40 mm bridge would add ~2.7 mm/s (~13%).
    assert abs(moved - still) / still < 0.03


def test_off_template_catches_circles_and_off_centre_but_not_honest_tremor():
    t, x, y = spiral(tremor_hz=6.0, pp_mm=1.0)
    assert "off_template" not in [r["code"] for r in metrics.analyze(t, x, y, template=TPL)["reasons"]]
    n = t.size
    k = n // 3
    radii = np.repeat([8.0, 16.0, 24.0], k)
    ph = np.tile(np.linspace(0, 2 * np.pi, k), 3)
    tc = t[: 3 * k]
    codes = [
        r["code"]
        for r in metrics.analyze(tc, radii * np.cos(ph), radii * np.sin(ph), template=TPL)["reasons"]
    ]
    assert "off_template" in codes
    # The same honest spiral drawn 6 mm away from the printed one.
    codes = [r["code"] for r in metrics.analyze(t, x + 6.0, y, template=TPL)["reasons"]]
    assert "off_template" in codes
    # Starting at the outer end is still honest.
    codes = [r["code"] for r in metrics.analyze(t, x[::-1], y[::-1], template=TPL)["reasons"]]
    assert "off_template" not in codes
