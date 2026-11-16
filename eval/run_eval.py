"""Spiral Test evaluation harness.

    uv run python eval/run_eval.py            # full run, writes results/
    uv run python eval/run_eval.py --quick    # smoke-size run (not for reporting)

Every random draw comes from numpy Generators seeded from SEED plus the experiment
and trial index, and every trial is a pure function of its spec, so the output is
identical regardless of worker count. Thresholds are chosen on the dev split only;
headline numbers are on the held-out split.

Data: UCI dataset 395 (Isenkul et al. 2014): Wacom tablet, pen, 15 controls and
62 Parkinson's subjects. No essential-tremor patients, no phone recordings.
"""

# ruff: noqa: E501  (report prose and table templates read better unwrapped)
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from spiral import metrics as M  # noqa: E402
from spiral import uci  # noqa: E402

SEED = 20261204
RESULTS = REPO / "results"
FIG = RESULTS / "fig"

# Pen lifts: UCI has no pen-up flag (pressure is never 0, Z is not a contact flag),
# but pressure falls to the bottom decile right before every gap > 50 ms, and the
# tablet only reports while the pen touches. So a gap > 50 ms (>= 5 missed samples)
# starts a new stroke. A backwards clock jump also starts one.
LIFT_GAP_MS = 50.0

FREQS_K = list(range(4, 13))  # 4..12 Hz, each with a U(-0.5, 0.5) offset, clipped to [4, 12]
AMPS = [0.25, 0.5, 1.0, 2.0, 4.0]  # peak-to-peak mm
GEOMS = ["linear", "radial", "elliptical"]
E1_REPS = 2
RATES = [60.0, 45.0, 30.0, 24.0, 20.0]
RATE_AMPS = [1.0, 2.0]
JITTER_MS = 2.0
QUANT_MM = 0.16  # ~1 CSS px on a phone
COMPRESS = [1.0, 1.5, 2.0, 3.0, 4.0]
E4_FREQ = 5.0
E4_PP = 1.0
ENV_DEPTH = 0.3  # envelope 1 + 0.3 sin(2 pi fe t), fe ~ U(0.1, 0.3) Hz
N_BOOT = 1000

BAR_FREQ_TOL = 0.5
BAR_FREQ_SHARE = 0.90
BAR_AMP_TOL = 0.25
BAR_AMP_SHARE = 0.90  # interpretation: >= 90% of trials within +-25% (median also reported)
FALSE_RHYTHM_MAX = 0.05
RATE_MARGIN = 0.95  # min_rate = 0.95 x lowest validated rate (timestamp noise in the measured rate)

# Values before this evaluation. max_speed 30 was the first run's dev result, superseded once
# MIN_RHYTHM_PP_MM existed (it rejected 51-57% of PD drawings).
OLD = {"MIN_PROMINENCE": 4.0, "MIN_RHYTHM_PP_MM": 0.0, "GATE": {"min_rate_hz": 30.0, "max_speed_mm_s": 60.0}}
AMP_MARGIN = (
    1.25  # MIN_RHYTHM_PP_MM = 1.25 x largest dev-control un-injected amplitude, rounded up to 0.05 mm
)
AMIN = 0.0  # set in main() from dev; read by rh()


def rh(r, tau):
    """'Dominant rhythm' as analyze() decides it: prominence AND amplitude."""
    return r["prom"] >= tau and r["amp"] >= AMIN


# ------------------------------------------------------------------ data


def load_records():
    recs = []
    for d in uci.load_all():
        t = np.asarray(d.t_ms, float)
        dt = np.diff(t)
        stroke = np.concatenate([[0], np.cumsum((dt > LIFT_GAP_MS) | (dt < 0))]).astype(int)
        flags = []
        if (dt < 0).any():
            flags.append("clock_reversal")
        recs.append(
            {
                "id": f"{d.subject}/t{d.test}",
                "subject": d.subject,
                "group": d.group,
                "source": d.source,
                "test": d.test,
                "t": t,
                "x": np.asarray(d.x_mm, float),
                "y": np.asarray(d.y_mm, float),
                "stroke": stroke,
                "flags": flags,
                "rate_nominal": float(1000.0 / np.median(dt[dt > 0])),
            }
        )
    seen = {}
    for r in recs:
        key = hashlib.sha1(np.stack([r["t"], r["x"], r["y"]]).tobytes()).hexdigest()
        if key in seen:
            other = seen[key]
            other["flags"].append(f"duplicate_of:{r['id']}")
            r["flags"].append(f"duplicate_of:{other['id']}")
        else:
            seen[key] = r
    return recs


def make_split(recs):
    rng = np.random.default_rng([SEED, 0])
    split = {}
    for group in ("control", "parkinson"):
        subs = sorted({r["subject"] for r in recs if r["group"] == group})
        perm = [subs[i] for i in rng.permutation(len(subs))]
        n_dev = len(subs) // 2
        split[group] = {"dev": sorted(perm[:n_dev]), "heldout": sorted(perm[n_dev:])}
    for r in recs:
        r["split"] = "dev" if r["subject"] in split[r["group"]]["dev"] else "heldout"
    return split


# ------------------------------------------------------------------ synthesis


def tremor(ts, spec, x, y):
    """Displacement (dx, dy) in mm at times ts (s from drawing start)."""
    env = 1.0 + ENV_DEPTH * np.sin(2 * np.pi * spec["env_f"] * ts + spec["env_ph"])
    ph = 2 * np.pi * spec["f"] * ts + spec["phase"]
    a = 0.5 * spec["pp"] * env
    ux, uy = math.cos(spec["axis"]), math.sin(spec["axis"])
    g = spec["geom"]
    if g == "linear":
        s = a * np.sin(ph)
        return s * ux, s * uy
    if g == "radial":
        cx, cy = spec["center"]
        r = np.hypot(x - cx, y - cy)
        r[r == 0] = 1.0
        s = a * np.sin(ph)
        return s * (x - cx) / r, s * (y - cy) / r
    if g == "elliptical":
        e = spec["ecc"]
        c, s = a * np.cos(ph), a * e * np.sin(ph)
        return c * ux - s * uy, c * uy + s * ux
    raise ValueError(g)


def synth(rec, spec):
    """Return (t_ms, x, y, stroke, true_pp) for one trial. Pure function of (rec, spec)."""
    t, x, y, s = rec["t"], rec["x"], rec["y"], rec["stroke"]
    k = spec.get("compress", 1.0)
    if k != 1.0:
        t = t[0] + (t - t[0]) / k
    rng = np.random.default_rng(spec["seed"])
    rate = spec.get("rate")
    if rate:
        tn = t[0] + np.arange(int((t[-1] - t[0]) * rate / 1000.0) + 1) * (1000.0 / rate)
        idx = np.clip(np.searchsorted(t, tn, side="right") - 1, 0, t.size - 2)
        inside = (t[idx + 1] - t[idx]) <= LIFT_GAP_MS * (1.0 / k)  # no samples while the pen is up
        tn, idx = tn[inside], idx[inside]
        x, y, s = np.interp(tn, t, x), np.interp(tn, t, y), s[idx]
        t_true = tn
        t_rec = tn + rng.uniform(-JITTER_MS, JITTER_MS, tn.size)
    else:
        t_true = t_rec = t
    true_pp = 0.0
    if spec.get("pp", 0) > 0:
        ts = (t_true - t_true[0]) / 1000.0
        dx, dy = tremor(ts, spec, x, y)
        x, y = x + dx, y + dy
        env = 1.0 + ENV_DEPTH * np.sin(2 * np.pi * spec["env_f"] * ts + spec["env_ph"])
        true_pp = float(spec["pp"] * np.median(env))
    if rate:
        x = np.round(x / QUANT_MM) * QUANT_MM
        y = np.round(y / QUANT_MM) * QUANT_MM
    return t_rec, x, y, s, true_pp


# ------------------------------------------------------------------ baselines


def radial_baseline(t, x, y, s, center):
    """Longardner et al. 2024 style: radial error vs a 2 Hz zero-phase low-pass of r.

    Frequency: PSD peak (4-12 Hz) of the radial residual. Amplitude: their
    mean |r - r_ideal|; converted to peak-to-peak as pi * mean|e| (exact for a sinusoid).
    """
    tc, xc, yc, _ = M.clean(t, x, y, s)
    _, xg, yg = M.resample(tc, xc, yc)
    r = np.hypot(xg - center[0], yg - center[1])
    e = r - M.lowpass(r, edge=(1.5, 2.5))
    trim = int(M.TRIM_S * M.FS)
    e_t = e[trim:-trim] if e.size > 2 * trim + int(M.FS) else e
    f, psd = M.welch(e_t)
    fr, prom = M.spectral_peak(f, psd)
    mean_err = float(np.mean(np.abs(e_t)))
    return {"rad_freq": fr, "rad_prom": prom, "rad_mean_err": mean_err, "rad_pp": math.pi * mean_err}


def zero_cross_freq(rx, ry):
    """Naive time-domain estimator: zero-crossing rate of the band-passed principal axis."""
    a, b = rx - rx.mean(), ry - ry.mean()
    cov = np.cov(np.stack([a, b]))
    w, v = np.linalg.eigh(cov)
    p = v[0, -1] * a + v[1, -1] * b
    zc = int(np.sum(np.signbit(p[1:]) != np.signbit(p[:-1])))
    return zc / (2.0 * (p.size / M.FS))


# ------------------------------------------------------------------ worker

_RECS = None


def _init_worker():
    global _RECS
    _RECS = load_records()


def run_trial(spec):
    rec = _RECS[spec["rec"]]
    t, x, y, s, true_pp = synth(rec, spec)
    m = M.compute(t, x, y, s)
    sp = m["sampling"]
    out = {
        "freq": m["freq_hz"],
        "prom": m["prominence"],
        "amp": m["amp_pp_mm"],
        "band_power": m["band_power_mm2"],
        "speed": m["speed_mm_s"],
        "rate": sp["rate_hz"],
        "max_gap": sp["max_gap_ms"],
        "clock_backsteps": sp.get("clock_backsteps", 0),
        "pen_time": sp.get("pen_time_s", 0.0),
        "lifts": m["lifts"],
        "true_pp": true_pp,
        "center": m["center"],
    }
    if spec.get("baselines"):
        out.update(radial_baseline(t, x, y, s, m["center"]))
        out["zc_freq"] = zero_cross_freq(m["series"]["rx"], m["series"]["ry"])
    return out


def gate_reasons(res, cfg):
    mini = {
        "sampling": {
            "rate_hz": res["rate"],
            "max_gap_ms": res["max_gap"],
            "pen_time_s": res["pen_time"],
            "clock_backsteps": res.get("clock_backsteps", 0),
        },
        "lifts": res["lifts"],
        "speed_mm_s": res["speed"],
    }
    return [r["code"] for r in M.gate(mini, cfg)]


# ------------------------------------------------------------------ spec builders


def inj_params(rng, fk, pp, geom):
    return {
        "f": float(np.clip(fk + rng.uniform(-0.5, 0.5), 4.0, 12.0)),
        "fk": fk,
        "pp": pp,
        "geom": geom,
        "phase": float(rng.uniform(0, 2 * np.pi)),
        "axis": float(rng.uniform(0, np.pi)),
        "ecc": float(rng.uniform(0.2, 1.0)),
        "env_f": float(rng.uniform(0.1, 0.3)),
        "env_ph": float(rng.uniform(0, 2 * np.pi)),
    }


def build_specs(recs, base, quick):
    specs = []
    ctrl = [i for i, r in enumerate(recs) if r["group"] == "control"]
    pd_ok = [i for i, r in enumerate(recs) if r["group"] == "parkinson" and not r["flags"]]
    freqs = [4, 8, 12] if quick else FREQS_K
    amps = [1.0, 2.0] if quick else AMPS
    reps = 1 if quick else E1_REPS
    ctrl_use = ctrl[:: (5 if quick else 1)]

    def add(exp, i, extra):
        n = len(specs)
        sp = {"exp": exp, "rec": i, "seed": [SEED, hash_exp(exp), n], **extra}
        specs.append(sp)

    # E0 / base: every drawing as recorded.
    for i in range(len(recs)):
        add("base", i, {"pp": 0.0, "baselines": True})
    # E1 + E5 (baselines on the same trials).
    for i in ctrl_use:
        for g in GEOMS:
            for fk in freqs:
                for pp in amps:
                    for rep in range(reps):
                        rng = np.random.default_rng([SEED, 1, i, GEOMS.index(g), fk, int(pp * 100), rep])
                        p = inj_params(rng, fk, pp, g)
                        p["center"] = base[i]["center"]
                        add("e1", i, {**p, "baselines": True})
    # E3: simulated phone sampling, injected + un-injected, controls; PD un-injected.
    for rate in RATES:
        for i in ctrl_use:
            for rep in range(3):
                add("e3_null", i, {"pp": 0.0, "rate": rate, "rep": rep})
            for g in GEOMS:
                for fk in freqs:
                    for pp in RATE_AMPS:
                        rng = np.random.default_rng(
                            [SEED, 3, int(rate), i, GEOMS.index(g), fk, int(pp * 100)]
                        )
                        p = inj_params(rng, fk, pp, g)
                        p["center"] = base[i]["center"]
                        add("e3", i, {**p, "rate": rate})
        if rate in (60.0, 30.0, 20.0):
            for i in pd_ok:
                add("e3_pd", i, {"pp": 0.0, "rate": rate})
    # E4: time compression.
    for k in COMPRESS:
        for i in ctrl_use:
            add("e4_null", i, {"pp": 0.0, "compress": k})
            for g in GEOMS:
                for rep in range(2):
                    rng = np.random.default_rng([SEED, 4, int(k * 10), i, GEOMS.index(g), rep])
                    p = inj_params(rng, E4_FREQ, E4_PP, g)
                    p["f"] = E4_FREQ
                    p["center"] = base[i]["center"]
                    add("e4", i, {**p, "compress": k})
    return specs


def hash_exp(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def run_all(specs, workers):
    if workers <= 1:
        _init_worker()
        return [run_trial(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        return list(ex.map(run_trial, specs, chunksize=16))


# ------------------------------------------------------------------ stats helpers


def pct(a):
    a = np.asarray(a, float)
    return float(100.0 * a.mean()) if a.size else float("nan")


def q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def auc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def boot_auc(neg, pos, key):
    rng = np.random.default_rng([SEED, 6, key])
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    vals = [
        auc(neg[rng.integers(0, neg.size, neg.size)], pos[rng.integers(0, pos.size, pos.size)])
        for _ in range(N_BOOT)
    ]
    return [q(vals, 2.5), q(vals, 97.5)]


def icc21(Y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm
    mse = (resid**2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def agreement(Y, key):
    """Y: n x 2 log amplitudes. ICC(2,1) with subject-bootstrap CI, SEM, MDC95 as % change."""
    n = Y.shape[0]
    icc = icc21(Y)
    rng = np.random.default_rng([SEED, 7, key])
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Yb = Y[idx]
        if np.ptp(Yb[:, 0]) == 0:
            continue
        boots.append(icc21(Yb))
    d = Y[:, 1] - Y[:, 0]
    sem = float(math.sqrt(np.sum(d * d) / (2 * n)))  # within-subject SD incl. task bias
    mdc = 1.96 * math.sqrt(2) * sem
    return {
        "n_subjects": n,
        "icc_2_1": icc,
        "icc_ci95": [q(boots, 2.5), q(boots, 97.5)],
        "sem_log": sem,
        "mdc95_log": mdc,
        "mdc95_pct_change": 100.0 * (math.exp(mdc) - 1.0),
        "mean_log_diff_dynamic_minus_static": float(d.mean()),
    }


def strict_ok(r, tau):
    return (rh(r, tau)) and abs(r["freq"] - r["spec"]["f"]) <= BAR_FREQ_TOL


def lenient_ok(r):
    return abs(r["freq"] - r["spec"]["f"]) <= BAR_FREQ_TOL


def amp_ok(r):
    return abs(r["amp"] / r["true_pp"] - 1.0) <= BAR_AMP_TOL


def recovery(rows, tau):
    """Summary of a set of injected trials."""
    if not rows:
        nan = float("nan")
        return {
            "n": 0,
            "freq_within_0p5_pct_reported": nan,
            "freq_within_0p5_pct_any_prominence": nan,
            "median_abs_freq_err_hz": nan,
            "detection_pct": nan,
            "amp_ratio_median": nan,
            "amp_ratio_iqr": [nan, nan],
            "amp_within_25pct_pct": nan,
        }
    ratio = np.array([r["amp"] / r["true_pp"] for r in rows])
    ferr = np.array([abs(r["freq"] - r["spec"]["f"]) for r in rows])
    return {
        "n": len(rows),
        "freq_within_0p5_pct_reported": pct([strict_ok(r, tau) for r in rows]),
        "freq_within_0p5_pct_any_prominence": pct([lenient_ok(r) for r in rows]),
        "median_abs_freq_err_hz": q(ferr, 50),
        "detection_pct": pct([rh(r, tau) for r in rows]),
        "amp_ratio_median": q(ratio, 50),
        "amp_ratio_iqr": [q(ratio, 25), q(ratio, 75)],
        "amp_within_25pct_pct": pct([amp_ok(r) for r in rows]),
    }


def bars(rows, tau):
    """Pre-registered bars on trials with pp >= 1 mm."""
    sel = [r for r in rows if r["spec"]["pp"] >= 1.0]
    if not sel:
        return {"n": 0, "freq_pass": None, "amp_pass": None}
    fr = pct([strict_ok(r, tau) for r in sel])
    am = pct([amp_ok(r) for r in sel])
    return {
        "n": len(sel),
        "freq_pct": fr,
        "amp_pct": am,
        "freq_pass": bool(fr >= 100 * BAR_FREQ_SHARE),
        "amp_pass": bool(am >= 100 * BAR_AMP_SHARE),
    }


# ------------------------------------------------------------------ experiments


def e2_threshold(base_rows):
    dev = [r["prom"] for r in base_rows if r["rec"]["group"] == "control" and r["rec"]["split"] == "dev"]
    grid = np.round(np.arange(1.0, 100.0001, 0.25), 2)
    for tau in grid:
        if np.mean(np.array(dev) >= tau) <= FALSE_RHYTHM_MAX:
            return float(tau), dev
    return float(grid[-1]), dev


def e3_min_rate(e3_rows, tau):
    per_rate = {}
    for rate in RATES:
        rows = [r for r in e3_rows if r["spec"]["rate"] == rate and r["rec"]["split"] == "dev"]
        per_rate[rate] = bars(rows, tau)
    validated = None
    for rate in sorted(RATES, reverse=True):  # walk down from the highest rate
        b = per_rate[rate]
        if b["freq_pass"] and b["amp_pass"]:
            validated = rate
        else:
            break
    return validated, per_rate


def e4_max_speed(e4_null, tau, keep):
    """Largest S such that, at every S' <= S, dev false rhythm (prominence + amplitude rules)
    among time-compressed un-injected controls with speed <= S' that pass every other gate
    check is <= 5%. Capped at the fastest such dev drawing (no extrapolation)."""
    nul = sorted((r for r in e4_null if r["rec"]["split"] == "dev" and keep(r)), key=lambda r: r["speed"])
    curve, best = [], None
    for hi in np.arange(10.0, 300.0001, 5.0):
        sub = [r for r in nul if r["speed"] <= hi]
        if not sub:
            continue
        fa = pct([rh(r, tau) for r in sub])
        curve.append({"max_speed": float(hi), "n": len(sub), "false_rhythm_pct": fa})
        if fa > 100 * FALSE_RHYTHM_MAX:
            break
        best = float(min(hi, 5 * math.floor(nul[-1]["speed"] / 5))) if len(sub) == len(nul) else float(hi)
        if len(sub) == len(nul):
            break
    return best, curve


# ------------------------------------------------------------------ 11 Hz investigation


def investigate_11hz(recs):
    out = {"drawings": [], "synthetic": []}
    for r in recs:
        if "clock_reversal" not in r["flags"]:
            continue
        t, x, y, s = r["t"], r["x"], r["y"], r["stroke"]
        m = M.compute(t, x, y, s)
        # The pre-fix behaviour: clean() used to sort by time. Sorting here reproduces it exactly.
        o = np.argsort(t, kind="stable")
        m_old = M.compute(t[o], x[o], y[o], s[o])
        j = int(np.where(np.diff(t) < 0)[0][0])
        overlap = int(np.sum(t[j + 1 :] <= t[j]))
        # Repair: keep file order, re-time the block after the jump to follow the block before it.
        t2 = t.copy()
        t2[j + 1 :] += (t[j] - t[j + 1]) + np.median(np.diff(t[: j + 1]))
        m2 = M.compute(t2, x, y, s)
        g = [c["code"] for c in M.gate(m)]
        out["drawings"].append(
            {
                "id": r["id"],
                "nominal_rate_hz": r["rate_nominal"],
                "clock_jump_ms": float(t[j + 1] - t[j]),
                "samples_interleaved_after_sort": overlap,
                "old_sorting_clean": {
                    "freq_hz": m_old["freq_hz"],
                    "prominence": m_old["prominence"],
                    "amp_pp_mm": m_old["amp_pp_mm"],
                    "rms_mm": m_old["rms_mm"],
                    "gate": [c["code"] for c in M.gate(m_old)],
                },
                "as_recorded": {
                    "freq_hz": m["freq_hz"],
                    "prominence": m["prominence"],
                    "amp_pp_mm": m["amp_pp_mm"],
                    "rms_mm": m["rms_mm"],
                    "clock_backsteps": m["sampling"]["clock_backsteps"],
                    "gate": g,
                },
                "file_order_repaired": {
                    "freq_hz": m2["freq_hz"],
                    "prominence": m2["prominence"],
                    "amp_pp_mm": m2["amp_pp_mm"],
                    "rms_mm": m2["rms_mm"],
                },
            }
        )
    # Synthetic: a smooth 25 s spiral, every other sample displaced 1 mm (two interleaved
    # trajectories), at the tablet's two clock periods.
    for dt_ms in (9.0, 7.0):
        n = int(25000 / dt_ms)
        t = np.arange(n) * dt_ms
        th = np.linspace(0, 6 * np.pi, n)
        x = (5 + 3 * th) * np.cos(th)
        y = (5 + 3 * th) * np.sin(th)
        x = x + (np.arange(n) % 2) * 1.0
        m = M.compute(t, x, y)
        out["synthetic"].append(
            {
                "sample_period_ms": dt_ms,
                "sample_rate_hz": 1000 / dt_ms,
                "predicted_alias_hz": abs(1000 / dt_ms - M.FS),
                "freq_hz": m["freq_hz"],
                "prominence": m["prominence"],
            }
        )
    return out


# ------------------------------------------------------------------ E10 template gate (synthetic)

PHONE_TPL = {"cx": 0.0, "cy": 0.0, "a": 4.0, "b": 26.0 / (6 * math.pi), "turns": 3.0, "phi0": 0.0, "dir": 1}
TPL_KINDS = [
    "honest",
    "scribble",
    "concentric_circles",
    "off_centre_spiral",
    "reversed_spiral",
    "wrong_pitch_spiral",
]


def _smooth_noise(rng, n, fs, sd, cutoff=(0.3, 0.6)):
    z = M.lowpass(rng.normal(0, 1, n), fs=fs, edge=cutoff)
    return sd * z / (z.std() or 1.0)


def synth_template_trace(kind, rng):
    """A phone-like trace (60 Hz, +-2 ms jitter, 0.16 mm quantisation) over PHONE_TPL."""
    fs = 60.0
    dur = rng.uniform(10, 30)
    n = int(dur * fs)
    t = np.arange(n) / fs
    a, b = PHONE_TPL["a"], PHONE_TPL["b"]
    u = t / dur
    u = u + 0.03 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * u) * u * (1 - u)  # uneven pace
    th0, th1 = rng.uniform(0, 0.5), 6 * np.pi - rng.uniform(0, 0.5)
    th = th0 + (th1 - th0) * u
    if kind == "honest" and rng.uniform() < 0.15:
        th = th[::-1]  # some people start at the outer end
    r = a + b * th + rng.uniform(-1.0, 1.0) + _smooth_noise(rng, n, fs, rng.uniform(0, 1.5))
    # kn: honest traces never cross the centre here; one that does loses turns (coverage), untested on people.
    r = np.maximum(r, 1.5)
    cx = cy = 0.0
    if kind == "off_centre_spiral":
        ang = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(4, 10)
        cx, cy = off * math.cos(ang), off * math.sin(ang)
    if kind == "wrong_pitch_spiral":
        r = a + b * rng.choice([0.6, 1.6]) * th
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    if kind == "reversed_spiral":
        y = -y
    if kind == "concentric_circles":
        k = int(rng.integers(3, 5))
        radii = np.sort(rng.uniform(5, 28, k))
        R = np.repeat(radii, int(np.ceil(n / k)))[:n] + _smooth_noise(rng, n, fs, 0.5)
        ph = np.tile(np.linspace(0, 2 * np.pi, int(np.ceil(n / k))), k)[:n] + rng.uniform(0, 2 * np.pi)
        x, y = R * np.cos(ph), R * np.sin(ph)
    if kind == "scribble":
        x = np.cumsum(rng.normal(0, 0.6, n)) + rng.uniform(-10, 10)
        y = np.cumsum(rng.normal(0, 0.6, n)) + rng.uniform(-10, 10)
    pp, f, ax = rng.uniform(0, 4), rng.uniform(4, 12), rng.uniform(0, np.pi)
    d = 0.5 * pp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x, y = x + d * math.cos(ax), y + d * math.sin(ax)
    x, y = np.round(x / QUANT_MM) * QUANT_MM, np.round(y / QUANT_MM) * QUANT_MM
    t_ms = t * 1000 + rng.uniform(-JITTER_MS, JITTER_MS, n)
    return t_ms, x, y


def e10_template(quick):
    """Set max_template_dev_mm on synthetic dev traces; report on separately seeded held-out traces."""
    n_h, n_a = (40, 8) if quick else (300, 60)
    out = {}
    rows = {}
    for split, sk in (("dev", 1), ("heldout", 2)):
        rows[split] = []
        for kind in TPL_KINDS:
            for i in range(n_h if kind == "honest" else n_a):
                rng = np.random.default_rng([SEED, 10, sk, TPL_KINDS.index(kind), i])
                tr = synth_template_trace(kind, rng)
                m = M.compute(*tr, template=PHONE_TPL)
                rows[split].append(
                    {
                        "kind": kind,
                        "dev": m["template_dev_mm"],
                        "old": old_nearest_arm(tr[1], tr[2], PHONE_TPL),
                        "m": m,
                    }
                )
    honest_dev = [r["dev"] for r in rows["dev"] if r["kind"] == "honest"]
    thr = float(math.ceil(np.percentile(honest_dev, 99) / 0.1 - 1e-9) * 0.1)
    for split in ("dev", "heldout"):
        d = {}
        for kind in TPL_KINDS:
            rr = [r for r in rows[split] if r["kind"] == kind]
            gates = [[c["code"] for c in M.gate(r["m"], {"max_template_dev_mm": thr})] for r in rr]
            d[kind] = {
                "n": len(rr),
                "template_dev_median_mm": q([r["dev"] for r in rr], 50),
                "template_dev_p99_mm": q([r["dev"] for r in rr], 99),
                "off_template_pct": pct([r["dev"] > thr for r in rr]),
                "any_template_gate_pct": pct([("off_template" in g) or ("coverage" in g) for g in gates]),
                "old_nearest_arm_off_template_pct": pct([r["old"] > OLD_TEMPLATE_THR for r in rr]),
            }
        out[split] = d
    return thr, out


OLD_TEMPLATE_THR = 10.0


def old_nearest_arm(x, y, tpl):
    """The previous template_deviation (RMS distance to the nearest arm), kept for comparison only."""
    ang = np.arctan2(y - tpl["cy"], x - tpl["cx"])
    alpha = np.mod(tpl.get("dir", 1) * (ang - tpl.get("phi0", 0.0)), 2 * np.pi)
    r = np.hypot(x - tpl["cx"], y - tpl["cy"])
    k = np.arange(int(math.ceil(tpl["turns"])) + 1)[:, None]
    d = np.min(np.abs(r[None, :] - (tpl["a"] + tpl["b"] * (alpha[None, :] + 2 * np.pi * k))), axis=0)
    return float(math.sqrt(np.mean(d * d)))


# ------------------------------------------------------------------ E9 mutations


@contextmanager
def patched(name, value):
    old = getattr(M, name)
    setattr(M, name, value)
    try:
        yield
    finally:
        setattr(M, name, old)


def _identity_bandpass(sig, fs=M.FS, hp=M.HP, lp=M.LP):
    return np.asarray(sig, float).copy()


def _uniform_resample(t_ms, *sigs, fs=M.FS):
    n = len(t_ms)
    return (np.arange(n) / fs, *[np.asarray(s, float) for s in sigs])


_ORIG_SPECTRAL_PEAK = M.spectral_peak


def _peak_no_edge_rule(freqs, psd, band=M.BAND):
    inb = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    i = int(inb[np.argmax(psd[inb])])
    f_peak, _ = _ORIG_SPECTRAL_PEAK(freqs, psd, band)
    others = psd[inb][np.abs(freqs[inb] - freqs[i]) > 1.0]
    floor = float(np.median(others)) if others.size else float("nan")
    return f_peak, float(psd[i] / floor) if floor > 0 else float("inf")


def mutation_checks(recs, base, tau, min_rate):
    """Named checks, run in-process so monkeypatches apply. Each must be green unmutated."""
    _init_worker()
    held_ctrl = [i for i, r in enumerate(recs) if r["group"] == "control" and r["split"] == "heldout"]
    pick = held_ctrl[::4][:4]

    def inj_trials():
        rows = []
        for i in pick:
            for fk in (5, 8, 11):
                rng = np.random.default_rng([SEED, 9, i, fk])
                p = inj_params(rng, fk, 2.0, "linear")
                p["center"] = base[i]["center"]
                spec = {"rec": i, "seed": [SEED, 9, i, fk], **p}
                res = run_trial(spec)
                rows.append({**res, "spec": spec})
        return rows

    def c_freq():
        rows = inj_trials()
        v = pct([strict_ok(r, tau) for r in rows])
        return v >= 90.0, f"{v:.0f}% reported within 0.5 Hz (n={len(rows)}, need >=90%)"

    def c_amp():
        rows = inj_trials()
        v = q([r["amp"] / r["true_pp"] for r in rows], 50)
        return abs(v - 1) <= 0.25, f"median amp ratio {v:.2f} (need 0.75-1.25)"

    def c_spec():
        rows = []
        for i in held_ctrl:
            for k in (1.0, 2.0):
                r = run_trial({"rec": i, "seed": [SEED, 9, i], "pp": 0.0, "compress": k})
                if not M.gate(_mini(r)):
                    rows.append(r)
        v = pct([rh(r, tau) for r in rows])
        return (
            v <= 10.0,
            f"false rhythm {v:.0f}% on gate-passing un-injected controls x1/x2 (n={len(rows)}, need <=10%)",
        )

    def c_gate():
        rows = [run_trial({"rec": i, "seed": [SEED, 9, i], "pp": 0.0, "rate": 20.0}) for i in pick]
        ab = [("low_rate" in [c["code"] for c in M.gate(_mini(r))]) for r in rows]
        v = pct(ab)
        return v == 100.0, f"{v:.0f}% of 20 Hz inputs abstain with low_rate (need 100%)"

    def c_spec_x1_ungated():
        rows = [run_trial({"rec": i, "seed": [SEED, 9, i], "pp": 0.0}) for i in held_ctrl]
        v = pct([rh(r, tau) for r in rows])
        return (
            v <= 10.0,
            f"false rhythm {v:.0f}% on all un-injected controls, no gate (n={len(rows)}, need <=10%)",
        )

    checks = {
        "freq_recovery": c_freq,
        "amp_recovery": c_amp,
        "specificity": c_spec,
        "rate_gate": c_gate,
        "specificity_ungated (added post hoc)": c_spec_x1_ungated,
    }
    mutations = {
        "none": None,
        "bandpass_identity": ("bandpass", _identity_bandpass),
        "skip_resampling": ("resample", _uniform_resample),
        "no_band_edge_rejection": ("spectral_peak", _peak_no_edge_rule),
        "no_gate_rate_check": ("GATE", {**M.GATE, "min_rate_hz": 0.0}),
    }
    expected = {
        "bandpass_identity": "amp_recovery",
        "skip_resampling": "freq_recovery",
        "no_band_edge_rejection": "specificity",
        "no_gate_rate_check": "rate_gate",
    }
    table = {}
    for mname, mut in mutations.items():
        row = {}
        for cname, fn in checks.items():
            if mut is None:
                ok, msg = fn()
            else:
                with patched(*mut):
                    ok, msg = fn()
            row[cname] = {"green": bool(ok), "detail": msg}
        table[mname] = row
    verdict = {}
    for mname, cname in expected.items():
        verdict[mname] = {
            "expected_red": cname,
            "caught": not table[mname][cname]["green"],
            "red_checks": [c for c, v in table[mname].items() if not v["green"]],
        }
    return {
        "table": table,
        "verdict": verdict,
        "unmutated_all_green": all(v["green"] for v in table["none"].values()),
    }


def _mini(res):
    return {
        "sampling": {
            "rate_hz": res["rate"],
            "max_gap_ms": res["max_gap"],
            "pen_time_s": res["pen_time"],
            "clock_backsteps": res.get("clock_backsteps", 0),
        },
        "lifts": res["lifts"],
        "speed_mm_s": res["speed"],
    }


# ------------------------------------------------------------------ figures


PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def figures(R, e1_held, e3_held, base_rows, e7_pairs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    tau = R["thresholds"]["MIN_PROMINENCE"]["new"]

    # 1. E1 held-out: reported-frequency success vs frequency, per amplitude, per geometry.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, g in zip(axes, GEOMS, strict=True):
        for c, pp in zip(PAL, AMPS, strict=True):
            ys = []
            for fk in FREQS_K:
                rows = [
                    r
                    for r in e1_held
                    if r["spec"]["geom"] == g and r["spec"]["pp"] == pp and r["spec"]["fk"] == fk
                ]
                ys.append(pct([strict_ok(r, tau) for r in rows]))
            ax.plot(FREQS_K, ys, "-o", color=c, lw=2, ms=4, label=f"{pp:g} mm")
        ax.axhline(90, color=INK2, lw=1, ls="--")
        ax.set_title(f"{g} tremor", fontsize=10, color=INK)
        ax.set_xlabel("injected frequency (Hz)", fontsize=9, color=INK2)
        _style(ax)
    axes[0].set_ylabel("% reported within ±0.5 Hz", fontsize=9, color=INK2)
    axes[-1].legend(
        title="peak-to-peak",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    fig.suptitle(
        "E1 injection recovery, held-out controls (Wacom pen, real drawings + synthetic tremor); "
        "dashed = 90% bar; lines for 1, 2, 4 mm overlap at 100%",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIG / "e1_recovery.png", dpi=150)
    plt.close(fig)

    # 2. E3: success and amplitude ratio vs sampling rate.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    groups = [("4-6 Hz", (4, 6)), ("7-9 Hz", (7, 9)), ("10-12 Hz", (10, 12))]
    rates_all = [None, *RATES]
    xs = [140, *RATES]
    for c, (lab, (lo, hi)) in zip(PAL[:3], groups, strict=True):
        ys, ar = [], []
        for rate in rates_all:
            src = e1_held if rate is None else [r for r in e3_held if r["spec"]["rate"] == rate]
            rows = [r for r in src if lo <= r["spec"]["fk"] <= hi and r["spec"]["pp"] >= 1.0]
            ys.append(pct([strict_ok(r, tau) for r in rows]))
            ar.append(q([r["amp"] / r["true_pp"] for r in rows], 50))
        axes[0].plot(xs, ys, "-o", color=c, lw=2, ms=4, label=lab)
        axes[1].plot(xs, ar, "-o", color=c, lw=2, ms=4, label=lab)
    mr = R["thresholds"]["GATE.min_rate_hz"]["new"]
    for ax in axes:
        ax.axvline(mr, color=INK2, lw=1, ls=":")
        ax.set_xlabel("sampling rate (Hz); 140 = native tablet", fontsize=9, color=INK2)
        _style(ax)
    axes[0].axhline(90, color=INK2, lw=1, ls="--")
    axes[0].set_ylabel("% reported within ±0.5 Hz", fontsize=9, color=INK2)
    axes[1].axhspan(0.75, 1.25, color=GRID, alpha=0.6)
    axes[1].set_ylabel("median amplitude ratio (measured/true)", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        f"E3 simulated phone sampling (not a phone), held-out, pp ≥ 1 mm; dotted = gate min_rate {mr:g} Hz",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIG / "e3_sampling_rate.png", dpi=150)
    plt.close(fig)

    # 3. Prominence of un-injected drawings.
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cats = [
        ("controls dev", "control", "dev"),
        ("controls held-out", "control", "heldout"),
        ("PD dev", "parkinson", "dev"),
        ("PD held-out", "parkinson", "heldout"),
    ]
    rng = np.random.default_rng([SEED, 99])
    for k, (_lab, grp, spl) in enumerate(cats):
        v = np.array(
            [
                max(r["prom"], 0.3)
                for r in base_rows
                if r["rec"]["group"] == grp and r["rec"]["split"] == spl and not r["rec"]["flags"]
            ]
        )
        ax.scatter(
            v,
            k + rng.uniform(-0.2, 0.2, v.size),
            s=14,
            color=PAL[0 if grp == "control" else 1],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvline(4.0, color=INK2, ls=":", lw=1)
    ax.axvline(tau, color=INK, ls="--", lw=1)
    ax.set_ylim(-1.0, len(cats) - 0.2)
    ax.text(tau, -0.75, f" new {tau:g}", fontsize=8, color=INK)
    ax.text(4.0, -0.75, "old 4.0 ", fontsize=8, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_yticks(range(len(cats)), [c[0] for c in cats])
    ax.set_xlabel("spectral prominence (peak / median in-band PSD); 0 plotted at 0.3", fontsize=9, color=INK2)
    ax.set_title(
        "E2/E6 un-injected UCI drawings (tablet pen); corrupted/duplicate files excluded",
        fontsize=9,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "prominence.png", dpi=150)
    plt.close(fig)

    # 4. E7 static vs dynamic.
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for grp, c in (("control", PAL[0]), ("parkinson", PAL[1])):
        P = np.array([p[1:3] for p in e7_pairs if p[0] == grp], float)
        if P.size:
            ax.scatter(
                P[:, 0],
                P[:, 1],
                s=18,
                color=c,
                edgecolor="white",
                linewidth=0.5,
                label="controls" if grp == "control" else "Parkinson's",
            )
    lim = [0.08, 10]
    ax.plot(lim, lim, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("static spiral amplitude (mm pp)", fontsize=9, color=INK2)
    ax.set_ylabel("dynamic spiral amplitude (mm pp)", fontsize=9, color=INK2)
    ax.set_title("E7 between-task agreement (not test-retest)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "e7_agreement.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke tests; not for reporting")
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    t0 = time.time()

    recs = load_records()
    split = make_split(recs)

    # Base pass first (radial injection needs each drawing's fitted centre).
    base_specs = [
        {"exp": "base", "rec": i, "seed": [SEED, 0, i], "pp": 0.0, "baselines": True}
        for i in range(len(recs))
    ]
    base_res = run_all(base_specs, args.workers)
    specs = [s for s in build_specs(recs, base_res, args.quick) if s["exp"] != "base"]
    res = run_all(specs, args.workers)
    rows = [{**r, "spec": s, "rec": recs[s["rec"]]} for s, r in zip(specs, res, strict=True)]
    base_rows = [{**r, "spec": s, "rec": recs[s["rec"]]} for s, r in zip(base_specs, base_res, strict=True)]
    by = {}
    for r in rows:
        by.setdefault(r["spec"]["exp"], []).append(r)
    t_trials = time.time() - t0

    R = {"meta": {}, "config": {}, "thresholds": {}, "experiments": {}}

    # ---- E2 threshold (dev)
    tau, dev_proms = e2_threshold(base_rows)
    clean = lambda rr: [r for r in rr if not r["rec"]["flags"]]  # noqa: E731
    ctrl_base = {
        s: [r for r in base_rows if r["rec"]["group"] == "control" and r["rec"]["split"] == s]
        for s in ("dev", "heldout")
    }

    # ---- MIN_RHYTHM_PP_MM (dev): largest un-injected dev-control amplitude x 1.25, up to 0.05 mm
    global AMIN
    dev_amps = [r["amp"] for r in ctrl_base["dev"] if not r["rec"]["flags"]]
    AMIN = float(math.ceil(max(dev_amps) * AMP_MARGIN / 0.05 - 1e-9) * 0.05)

    # ---- E3 rate (dev)
    validated_rate, rate_bars_dev = e3_min_rate(by["e3"], tau)
    min_rate = math.floor(RATE_MARGIN * validated_rate) if validated_rate else None

    # ---- E4 speed (dev): false rhythm on compressed controls, conditional on every other gate check.
    pre_cfg = {"min_rate_hz": float(min_rate) if min_rate else M.GATE["min_rate_hz"]}

    def other_gates_ok(r):
        return not [c for c in gate_reasons(r, pre_cfg) if c != "too_fast"]

    max_speed, speed_curve = e4_max_speed(by["e4_null"], tau, other_gates_ok)
    max_speed_uncond, speed_curve_uncond = e4_max_speed(by["e4_null"], tau, lambda r: True)

    # ---- E10 template gate (synthetic only)
    tpl_thr, E10 = e10_template(args.quick)

    gate_cfg = {
        "min_rate_hz": float(min_rate) if min_rate else M.GATE["min_rate_hz"],
        "max_speed_mm_s": max_speed if max_speed else M.GATE["max_speed_mm_s"],
        "max_template_dev_mm": tpl_thr,
    }
    file_vals = {
        "MIN_PROMINENCE": M.MIN_PROMINENCE,
        "MIN_RHYTHM_PP_MM": M.MIN_RHYTHM_PP_MM,
        "min_rate_hz": M.GATE["min_rate_hz"],
        "max_speed_mm_s": M.GATE["max_speed_mm_s"],
        "max_template_dev_mm": M.GATE["max_template_dev_mm"],
    }
    R["thresholds"] = {
        "MIN_PROMINENCE": {
            "old": OLD["MIN_PROMINENCE"],
            "new": tau,
            "in_metrics_py": file_vals["MIN_PROMINENCE"],
            "rule": "smallest value on a 0.25 grid with <=5% of dev un-injected control drawings "
            "at or above it",
            "dev_control_prominences": sorted(dev_proms),
        },
        "MIN_RHYTHM_PP_MM": {
            "old": OLD["MIN_RHYTHM_PP_MM"],
            "new": AMIN,
            "in_metrics_py": file_vals["MIN_RHYTHM_PP_MM"],
            "rule": "1.25 x the largest pp amplitude of any dev un-injected control drawing, rounded up to "
            "0.05 mm (mm via the assumed 0.204 mm/px tablet scale)",
            "dev_control_max_amp_mm": max(dev_amps),
        },
        "GATE.min_rate_hz": {
            "old": OLD["GATE"]["min_rate_hz"],
            "new": gate_cfg["min_rate_hz"],
            "in_metrics_py": file_vals["min_rate_hz"],
            "rule": "lowest simulated rate at which both E1 bars hold on dev (and at every higher "
            "tested rate), x0.95 for timestamp noise in the measured rate",
            "validated_rate_hz": validated_rate,
            "dev_bars_by_rate": {str(k): v for k, v in rate_bars_dev.items()},
        },
        "GATE.max_template_dev_mm": {
            "old": OLD_TEMPLATE_THR,
            "new": tpl_thr,
            "in_metrics_py": file_vals["max_template_dev_mm"],
            "rule": "99th percentile of template_dev on synthetic honest dev traces, rounded up to 0.1 mm. "
            "SYNTHETIC ONLY: no UCI drawing has a template; untested on real phone drawings",
        },
        "GATE.max_speed_mm_s": {
            "old": OLD["GATE"]["max_speed_mm_s"],
            "new": gate_cfg["max_speed_mm_s"],
            "in_metrics_py": file_vals["max_speed_mm_s"],
            "rule": "largest S such that dev false rhythm (prominence AND amplitude rules) among time-compressed "
            "un-injected control drawings passing every other gate check, with speed <= S' for every S' <= S, "
            "stays <= 5%; capped at the fastest such drawing; 5 mm/s grid",
            "first_run_value": 30.0,
            "first_run_rule": "20 mm/s windows of injected+un-injected trials, prominence rule only (rejected "
            "51-57% of PD drawings; superseded on orchestrator instruction)",
            "dev_curve": speed_curve,
            "unconditional_value": max_speed_uncond,
            "unconditional_dev_curve": speed_curve_uncond,
        },
    }
    R["thresholds_match_metrics_py"] = (
        abs(file_vals["MIN_PROMINENCE"] - tau) < 1e-9
        and abs(file_vals["MIN_RHYTHM_PP_MM"] - AMIN) < 1e-9
        and abs(file_vals["min_rate_hz"] - gate_cfg["min_rate_hz"]) < 1e-9
        and abs(file_vals["max_speed_mm_s"] - gate_cfg["max_speed_mm_s"]) < 1e-9
        and abs(file_vals["max_template_dev_mm"] - tpl_thr) < 1e-9
    )

    # ---- E1
    e1 = by["e1"]
    E1 = {}
    for s in ("dev", "heldout"):
        part = [r for r in e1 if r["rec"]["split"] == s]
        E1[s] = {
            "by_geometry_amplitude": {
                g: {
                    f"{pp:g}mm": recovery(
                        [r for r in part if r["spec"]["geom"] == g and r["spec"]["pp"] == pp], tau
                    )
                    for pp in AMPS
                }
                for g in GEOMS
            },
            "bars_pp_ge_1mm": {g: bars([r for r in part if r["spec"]["geom"] == g], tau) for g in GEOMS},
            "bars_pp_ge_1mm_all_geometries": bars(part, tau),
            "by_frequency_pp_ge_1mm": {
                str(fk): recovery([r for r in part if r["spec"]["fk"] == fk and r["spec"]["pp"] >= 1.0], tau)
                for fk in FREQS_K
            },
            "n_drawings": len({r["rec"]["id"] for r in part}),
        }
    R["experiments"]["E1_injection_recovery"] = E1
    e1_held = [r for r in e1 if r["rec"]["split"] == "heldout"]

    # ---- E2 report
    E2 = {"MIN_PROMINENCE": tau, "MIN_RHYTHM_PP_MM": AMIN, "old": OLD["MIN_PROMINENCE"]}
    for s in ("dev", "heldout"):
        pr = [r["prom"] for r in ctrl_base[s]]
        E2[s] = {
            "n_drawings": len(pr),
            "false_rhythm_pct_new": pct([rh(r, tau) for r in ctrl_base[s]]),
            "false_rhythm_n_new": sum(rh(r, tau) for r in ctrl_base[s]),
            "false_rhythm_n_prominence_only": int(np.sum(np.array(pr) >= tau)),
            "false_rhythm_pct_old": pct(np.array(pr) >= OLD["MIN_PROMINENCE"]),
            "max_prominence": float(np.max(pr)),
            "false_rhythm_freqs_hz": [round(r["freq"], 2) for r in ctrl_base[s] if rh(r, tau)],
        }
        nul60 = [r for r in by["e3_null"] if r["spec"]["rate"] == 60.0 and r["rec"]["split"] == s]
        E2[s]["false_rhythm_pct_sim60"] = pct([rh(r, tau) for r in nul60])
        E2[s]["n_sim60"] = len(nul60)
        part = [r for r in e1 if r["rec"]["split"] == s]
        E2[s]["detection_cost_by_amplitude"] = {
            f"{pp:g}mm": {
                "detect_pct_old": pct(
                    [r["prom"] >= OLD["MIN_PROMINENCE"] for r in part if r["spec"]["pp"] == pp]
                ),
                "detect_pct_prominence_only": pct([r["prom"] >= tau for r in part if r["spec"]["pp"] == pp]),
                "detect_pct_new": pct([rh(r, tau) for r in part if r["spec"]["pp"] == pp]),
            }
            for pp in AMPS
        }
    R["experiments"]["E2_specificity"] = E2
    R["experiments"]["E10_template_gate_synthetic"] = {
        "label": "synthetic phone-like traces over the phone template (a=4 mm, b=26/(6 pi) mm/rad, 3 turns): "
        "honest = template + constant offset U(-1,1) mm + slow radial wander SD U(0,1.5) mm (<0.6 Hz) + tremor "
        "pp U(0,4) mm at U(4,12) Hz, 15% traced from the outer end; 60 Hz, +-2 ms jitter, 0.16 mm quantisation. "
        "The honest-trace error model is an assumption, not measured on people.",
        "threshold_mm": tpl_thr,
        **E10,
    }

    # ---- E3 report
    E3 = {
        "label": "simulated phone sampling (uniform resample + ±2 ms jitter + 0.16 mm quantisation), not a phone",
        "min_rate_hz": gate_cfg["min_rate_hz"],
    }
    e3_held = [r for r in by["e3"] if r["rec"]["split"] == "heldout"]
    for rate in RATES:
        d = {}
        for s in ("dev", "heldout"):
            part = [r for r in by["e3"] if r["spec"]["rate"] == rate and r["rec"]["split"] == s]
            nul = [r for r in by["e3_null"] if r["spec"]["rate"] == rate and r["rec"]["split"] == s]
            ab = [bool(gate_reasons(r, gate_cfg)) and "low_rate" in gate_reasons(r, gate_cfg) for r in part]
            d[s] = {
                "bars_pp_ge_1mm": bars(part, tau),
                "by_geometry": {g: bars([r for r in part if r["spec"]["geom"] == g], tau) for g in GEOMS},
                "recovery_pp_ge_1mm": recovery(part, tau),
                "freq_pct_by_band": {
                    lab: pct([strict_ok(r, tau) for r in part if lo <= r["spec"]["fk"] <= hi])
                    for lab, (lo, hi) in (("4-6", (4, 6)), ("7-9", (7, 9)), ("10-12", (10, 12)))
                },
                "measured_rate_median_hz": q([r["rate"] for r in part], 50),
                "gate_abstains_low_rate_pct": pct(ab),
                "false_rhythm_pct_uninjected": pct([rh(r, tau) for r in nul]),
                "n_uninjected": len(nul),
            }
        E3[str(int(rate))] = d
    # PD drawings: agreement with native when a native rhythm exists, and abstention.
    base_by_rec = {r["rec"]["id"]: r for r in base_rows}
    for rate in (60.0, 30.0, 20.0):
        pdr = [r for r in by["e3_pd"] if r["spec"]["rate"] == rate]
        withr = [r for r in pdr if rh(base_by_rec[r["rec"]["id"]], tau)]
        agree = [abs(r["freq"] - base_by_rec[r["rec"]["id"]]["freq"]) <= 0.5 and rh(r, tau) for r in withr]
        E3[f"pd_{int(rate)}"] = {
            "n_drawings": len(pdr),
            "n_with_native_rhythm": len(withr),
            "rhythm_agrees_pct": pct(agree),
            "gate_abstains_low_rate_pct": pct(["low_rate" in gate_reasons(r, gate_cfg) for r in pdr]),
            "amp_log_ratio_median": q(
                [math.log(r["amp"] / base_by_rec[r["rec"]["id"]]["amp"]) for r in pdr], 50
            ),
        }
    R["experiments"]["E3_phone_sampling"] = E3

    # ---- E4 report
    E4 = {
        "freq_hz": E4_FREQ,
        "pp_mm": E4_PP,
        "max_speed_mm_s": gate_cfg["max_speed_mm_s"],
        "confound": "time compression also shortens the drawing (fewer Welch segments)",
    }
    for s in ("dev", "heldout"):
        E4[s] = {}
        for k in COMPRESS:
            inj = [r for r in by["e4"] if r["spec"]["compress"] == k and r["rec"]["split"] == s]
            nul = [r for r in by["e4_null"] if r["spec"]["compress"] == k and r["rec"]["split"] == s]
            fast = [("too_fast" in gate_reasons(r, gate_cfg)) for r in inj]
            other = [not [c for c in gate_reasons(r, gate_cfg) if c != "too_fast"] for r in inj]
            kept = [r for r, f in zip(inj, fast, strict=True) if not f]
            nkept = [r for r in nul if "too_fast" not in gate_reasons(r, gate_cfg)]
            E4[s][f"x{k:g}"] = {
                "n_injected": len(inj),
                "speed_median_mm_s": q([r["speed"] for r in inj], 50),
                "recovery_all": recovery(inj, tau),
                "false_rhythm_pct_all": pct([rh(r, tau) for r in nul]),
                "speed_gate_abstains_pct": pct(fast),
                "passes_other_gates_pct": pct(other),
                "recovery_kept": recovery(kept, tau),
                "false_rhythm_pct_kept": pct([rh(r, tau) for r in nkept]) if nkept else None,
                "n_null_kept": len(nkept),
                "n_null": len(nul),
            }
    # Held-out check of the speed gate: everything the full gate keeps, all compression factors pooled.
    hk_inj = [r for r in by["e4"] if r["rec"]["split"] == "heldout" and not gate_reasons(r, gate_cfg)]
    hk_nul = [r for r in by["e4_null"] if r["rec"]["split"] == "heldout" and not gate_reasons(r, gate_cfg)]
    hk_slow = [
        r for r in by["e4_null"] if r["rec"]["split"] == "heldout" and r["speed"] > gate_cfg["max_speed_mm_s"]
    ]
    E4["heldout_gate_kept_pooled"] = {
        "n_injected_kept": len(hk_inj),
        "recovery": recovery(hk_inj, tau),
        "n_uninjected_kept": len(hk_nul),
        "false_rhythm_pct_kept": pct([rh(r, tau) for r in hk_nul]),
        "n_uninjected_rejected_too_fast": len(hk_slow),
        "false_rhythm_pct_rejected_too_fast": pct([rh(r, tau) for r in hk_slow]),
    }
    R["experiments"]["E4_speed_stress"] = E4

    # ---- E5 baselines (paired, held-out E1 trials)
    E5 = {
        "method_radial": "Longardner et al. 2024 style radial error (2 Hz zero-phase low-pass of r); frequency "
        "from the PSD of the radial residual; pp = pi * mean|e|",
        "method_zero_cross": "zero-crossing rate of the band-passed principal-axis signal",
    }
    for g in [*GEOMS, "all"]:
        part = [r for r in e1_held if (g == "all" or r["spec"]["geom"] == g) and r["spec"]["pp"] >= 1.0]
        ours = np.array([lenient_ok(r) for r in part], float)
        rad = np.array([abs(r["rad_freq"] - r["spec"]["f"]) <= 0.5 for r in part], float)
        zc = np.array([abs(r["zc_freq"] - r["spec"]["f"]) <= 0.5 for r in part], float)
        ae_o = np.array([abs(math.log(r["amp"] / r["true_pp"])) for r in part])
        ae_r = np.array([abs(math.log(max(r["rad_pp"], 1e-6) / r["true_pp"])) for r in part])
        rng = np.random.default_rng([SEED, 5, GEOMS.index(g) if g in GEOMS else 9])
        idx = [rng.integers(0, len(part), len(part)) for _ in range(N_BOOT)]
        d_rad = [float(np.mean(ours[i] - rad[i])) * 100 for i in idx]
        d_zc = [float(np.mean(ours[i] - zc[i])) * 100 for i in idx]
        d_amp = [float(np.median(ae_r[i] - ae_o[i])) for i in idx]
        E5[g] = {
            "n_trials": len(part),
            "freq_within_0p5_pct": {"ours_2d": pct(ours), "radial": pct(rad), "zero_cross": pct(zc)},
            "paired_diff_ours_minus_radial_pct_points": {
                "est": float((ours - rad).mean() * 100),
                "ci95": [q(d_rad, 2.5), q(d_rad, 97.5)],
            },
            "paired_diff_ours_minus_zero_cross_pct_points": {
                "est": float((ours - zc).mean() * 100),
                "ci95": [q(d_zc, 2.5), q(d_zc, 97.5)],
            },
            "amp_ratio_median": {
                "ours_2d": q([r["amp"] / r["true_pp"] for r in part], 50),
                "radial": q([r["rad_pp"] / r["true_pp"] for r in part], 50),
            },
            "median_paired_abs_log_amp_err_radial_minus_ours": {
                "est": float(np.median(ae_r - ae_o)),
                "ci95": [q(d_amp, 2.5), q(d_amp, 97.5)],
            },
        }
    R["experiments"]["E5_baselines"] = E5

    # ---- E6 PD vs control (exploratory)
    feats = {
        "amp_pp_mm (log)": lambda r: math.log(r["amp"]),
        "band_power_mm2 (log)": lambda r: math.log(r["band_power"]),
        "duration_s": lambda r: r["pen_time"],
        "speed_mm_s": lambda r: r["speed"],
        "mean_radial_err_mm (Longardner)": lambda r: r["rad_mean_err"],
    }
    usable = clean(base_rows)

    def subj_table(rows_):
        tab = {}
        for r in rows_:
            key = (r["rec"]["group"], r["rec"]["subject"], r["rec"]["split"], r["rec"]["source"])
            tab.setdefault(key, []).append(r)
        return tab

    stab = subj_table(usable)
    E6 = {
        "label": "exploratory; 15 controls total; not diagnostic; one value per subject (mean over tests)",
        "excluded_drawings": [
            r["rec"]["id"] + ": " + ",".join(r["rec"]["flags"]) for r in base_rows if r["rec"]["flags"]
        ],
    }
    cohorts = {
        "heldout": lambda k: k[2] == "heldout",
        "dev": lambda k: k[2] == "dev",
        "all_subjects": lambda k: True,
        "all_controls_vs_new_dataset_PD_only (same ~142 Hz clock)": lambda k: (
            k[0] == "control" or k[3] == "new_dataset"
        ),
    }
    for ci, (cname, sel) in enumerate(cohorts.items()):
        keys = [k for k in stab if sel(k)]
        d = {
            "n_control": sum(k[0] == "control" for k in keys),
            "n_pd": sum(k[0] == "parkinson" for k in keys),
        }
        for fi, (fname, fn) in enumerate(feats.items()):
            neg = [np.mean([fn(r) for r in stab[k]]) for k in keys if k[0] == "control"]
            pos = [np.mean([fn(r) for r in stab[k]]) for k in keys if k[0] == "parkinson"]
            d[fname] = {"auc": auc(neg, pos), "ci95": boot_auc(neg, pos, ci * 10 + fi)}
        E6[cname] = d
    pd_draw = [r for r in usable if r["rec"]["group"] == "parkinson"]
    rhy = [r for r in pd_draw if rh(r, tau)]
    rh_ok = [r for r in rhy if not gate_reasons(r, gate_cfg)]
    hist = {f"{lo}-{lo + 1}": sum(lo <= r["freq"] < lo + 1 for r in rhy) for lo in range(4, 12)}
    E6["pd_dominant_rhythm"] = {
        "n_pd_drawings": len(pd_draw),
        "n_with_rhythm": len(rhy),
        "pct_with_rhythm": pct([rh(r, tau) for r in pd_draw]),
        "freq_median_hz": q([r["freq"] for r in rhy], 50),
        "freq_iqr_hz": [q([r["freq"] for r in rhy], 25), q([r["freq"] for r in rhy], 75)],
        "freq_histogram": hist,
        "n_with_rhythm_and_gate_pass": len(rh_ok),
        "n_gate_pass": sum(not gate_reasons(r, gate_cfg) for r in pd_draw),
        "control_drawings_with_rhythm": sum(rh(r, tau) for r in usable if r["rec"]["group"] == "control"),
        "n_control_drawings": sum(r["rec"]["group"] == "control" for r in usable),
    }
    E6["sampling_rate_confound"] = {
        f"{g}/{src}": {
            "median_nominal_rate_hz": q(
                [
                    r["rec"]["rate_nominal"]
                    for r in base_rows
                    if r["rec"]["group"] == g and r["rec"]["source"] == src
                ],
                50,
            ),
            "n_drawings": sum(r["rec"]["group"] == g and r["rec"]["source"] == src for r in base_rows),
        }
        for g, src in (("control", "hw_dataset"), ("parkinson", "hw_dataset"), ("parkinson", "new_dataset"))
    }
    R["experiments"]["E6_pd_vs_control_exploratory"] = E6

    # ---- E7 agreement
    pairs = {}
    for r in usable:
        pairs.setdefault((r["rec"]["group"], r["rec"]["subject"], r["rec"]["split"]), {})[
            r["rec"]["test"]
        ] = r["amp"]
    e7_pairs = [(k[0], v[0], v[1], k[2]) for k, v in pairs.items() if 0 in v and 1 in v]
    E7 = {
        "label": "between-task (static vs dynamic spiral, same session), not test-retest; a lower bound on "
        "test-retest reliability",
        "reference": "Elble & Ellenbogen 2017: tablet spiral pp amplitude "
        "ICC 0.97, MDC 51% of baseline over 2 days, 20 ET patients",
    }
    for ki, (cname, sel) in enumerate(
        {
            "heldout": lambda p: p[3] == "heldout",
            "all_subjects": lambda p: True,
            "parkinson_all": lambda p: p[0] == "parkinson",
            "control_all": lambda p: p[0] == "control",
        }.items()
    ):
        Y = np.log(np.array([[p[1], p[2]] for p in e7_pairs if sel(p)]))
        E7[cname] = agreement(Y, ki)
    R["experiments"]["E7_static_vs_dynamic"] = E7

    # ---- E8 gate on real data
    E8 = {
        "gate_cfg": {**M.GATE, **gate_cfg},
        "note": "no printed template exists for UCI, so coverage/off_template are not evaluated; strokes derived "
        "from gaps > 50 ms (see LIFT_GAP_MS)",
    }
    for g in ("control", "parkinson"):
        for s in ("dev", "heldout"):
            part = [r for r in base_rows if r["rec"]["group"] == g and r["rec"]["split"] == s]
            reasons = [gate_reasons(r, gate_cfg) for r in part]
            counts = {}
            for rr in reasons:
                for c in rr:
                    counts[c] = counts.get(c, 0) + 1
            E8[f"{g}/{s}"] = {
                "n_drawings": len(part),
                "abstain_pct": pct([bool(rr) for rr in reasons]),
                "n_abstain": sum(bool(rr) for rr in reasons),
                "reason_counts": counts,
            }
    E8["corrupted_drawings_gate"] = {
        r["rec"]["id"]: gate_reasons(r, gate_cfg) for r in base_rows if "clock_reversal" in r["rec"]["flags"]
    }
    R["experiments"]["E8_gate_real_data"] = E8

    # ---- 11 Hz investigation
    R["experiments"]["audit_11hz"] = investigate_11hz(recs)

    # ---- E9 mutations (constants patched to the dev-derived values for the duration of the checks)
    with patched("MIN_PROMINENCE", tau), patched("GATE", {**M.GATE, **gate_cfg}):
        R["experiments"]["E9_mutations"] = mutation_checks(recs, base_res, tau, gate_cfg["min_rate_hz"])

    # ---- config, failure map, headline
    R["config"] = {
        "seed": SEED,
        "lift_gap_ms": LIFT_GAP_MS,
        "freqs_hz": "k + U(-0.5,0.5) clipped to [4,12], k in " + str(FREQS_K),
        "amplitudes_pp_mm": AMPS,
        "geometries": GEOMS,
        "envelope": f"1 + {ENV_DEPTH} sin(2 pi fe t + psi), fe ~ U(0.1,0.3) Hz; true pp = pp x median(envelope)",
        "e1_reps": E1_REPS,
        "rates_hz": RATES,
        "rate_amps_pp_mm": RATE_AMPS,
        "jitter_ms": JITTER_MS,
        "quant_mm": QUANT_MM,
        "compress": COMPRESS,
        "e4_freq_hz": E4_FREQ,
        "e4_pp_mm": E4_PP,
        "n_boot": N_BOOT,
        "bars": {
            "freq_tol_hz": BAR_FREQ_TOL,
            "freq_share": BAR_FREQ_SHARE,
            "amp_tol": BAR_AMP_TOL,
            "amp_share": BAR_AMP_SHARE,
            "applies_to": "pp >= 1 mm, 4-12 Hz",
            "reported_frequency": "a trial succeeds only if prominence >= MIN_PROMINENCE (the product shows a "
            "frequency) AND |f - f_true| <= 0.5 Hz",
        },
        "mm_per_px_assumed": uci.MM_PER_PX,
        "split": split,
        "quick": args.quick,
    }
    R["meta"] = {
        "dataset": "UCI 395 (Isenkul et al. 2014), Wacom tablet + pen; 15 controls, 62 Parkinson's; no essential "
        "tremor, no phone recordings",
        "n_drawings": len(recs),
        "n_trials": len(rows) + len(base_rows),
        "runtime_s": None,
        "trial_time_s": round(t_trials, 1),
        "workers": args.workers,
        "metrics_py_sha1": hashlib.sha1((REPO / "spiral" / "metrics.py").read_bytes()).hexdigest(),
        "citations": {
            "ios_60hz_cap": "https://bugs.webkit.org/show_bug.cgi?id=261961",
            "elble_2017": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5618112/",
            "longardner_2024": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11324214/",
        },
    }
    R["failure_map"] = failure_map(R)
    R["headline"] = headline(R)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.quick:
        figures(R, e1_held, e3_held, base_rows, e7_pairs)
    R["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (args.out / "eval.json").write_text(json.dumps(_jsonable(R), indent=1, allow_nan=False))
    (args.out / "EVAL.md").write_text(render_md(R))
    print(
        f"wrote {args.out / 'eval.json'} and EVAL.md in {R['meta']['runtime_s']} s "
        f"({R['meta']['n_trials']} pipeline runs)"
    )
    if not R["thresholds_match_metrics_py"]:
        print("NOTE: dev-derived thresholds differ from spiral/metrics.py; see EVAL.md 'Threshold changes'.")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------ failure map / headline


def _cell(status, text, n):
    text = text if len(text) <= 40 else text[:39] + "…"
    return {"status": status, "text": text, "n": int(n)}


def failure_map(R):
    X = R["experiments"]
    mr = R["thresholds"]["GATE.min_rate_hz"]["new"]
    ET = _cell("untested", "no ET data exists (UCI has none)", 0)
    cols = [
        "Healthy controls (real, un-injected)",
        "Injected tremor 4–12 Hz on controls",
        "Parkinson's (real)",
        "Essential tremor",
    ]
    rows = []

    e2h = X["E2_specificity"]["heldout"]
    b = X["E1_injection_recovery"]["heldout"]["bars_pp_ge_1mm_all_geometries"]
    allpass = all(
        v["freq_pass"] and v["amp_pass"]
        for v in X["E1_injection_recovery"]["heldout"]["bars_pp_ge_1mm"].values()
    )
    pdr = X["E6_pd_vs_control_exploratory"]["pd_dominant_rhythm"]
    rows.append(
        {
            "condition": "Pen on Wacom tablet (real)",
            "cells": [
                _cell(
                    "measured",
                    f"false rhythm {e2h['false_rhythm_n_new']}/{e2h['n_drawings']} (held-out)",
                    e2h["n_drawings"],
                ),
                _cell(
                    "measured" if allpass else "fails",
                    f"±0.5 Hz in {b['freq_pct']:.0f}%, amp ±25% {b['amp_pct']:.0f}%",
                    b["n"],
                ),
                _cell(
                    "measured",
                    f"rhythm in {pdr['n_with_rhythm']}/{pdr['n_pd_drawings']}; exploratory",
                    pdr["n_pd_drawings"],
                ),
                ET,
            ],
        }
    )

    dc = X["E2_specificity"]["heldout"]["detection_cost_by_amplitude"]
    amin = X["E2_specificity"]["MIN_RHYTHM_PP_MM"]
    e1h = X["E1_injection_recovery"]["heldout"]["by_geometry_amplitude"]
    n_small = sum(e1h[g][k]["n"] for g in GEOMS for k in ("0.25mm", "0.5mm"))
    rows.append(
        {
            "condition": f"Small tremor 0.25–0.5 mm pp (tablet; floor {amin:g} mm)",
            "cells": [
                _cell("measured", "same drawings as tablet row", e2h["n_drawings"]),
                _cell(
                    "fails",
                    f"detected: 0.25 mm {dc['0.25mm']['detect_pct_new']:.0f}%, 0.5 mm {dc['0.5mm']['detect_pct_new']:.0f}%",
                    n_small,
                ),
                _cell("untested", "no amplitude ground truth", 0),
                ET,
            ],
        }
    )

    def rate_row(label, rate):
        d = X["E3_phone_sampling"][str(int(rate))]["heldout"]
        pd = X["E3_phone_sampling"].get(f"pd_{int(rate)}")
        if rate < mr:
            c1 = _cell(
                "abstains",
                f"gate abstains {d['gate_abstains_low_rate_pct']:.0f}% (<{mr:g} Hz)",
                d["n_uninjected"],
            )
            c2 = _cell(
                "abstains",
                f"abstains {d['gate_abstains_low_rate_pct']:.0f}%; raw ±0.5 Hz "
                f"{d['bars_pp_ge_1mm']['freq_pct']:.0f}%",
                d["bars_pp_ge_1mm"]["n"],
            )
            c3 = (
                _cell("abstains", f"gate abstains {pd['gate_abstains_low_rate_pct']:.0f}%", pd["n_drawings"])
                if pd
                else _cell("untested", "not simulated", 0)
            )
        else:
            bb = d["bars_pp_ge_1mm"]
            ok = bb["freq_pass"] and bb["amp_pass"]
            c1 = _cell(
                "measured",
                f"false rhythm {d['false_rhythm_pct_uninjected']:.0f}% (simulated)",
                d["n_uninjected"],
            )
            c2 = _cell(
                "measured" if ok else "fails",
                f"±0.5 Hz {bb['freq_pct']:.0f}%, amp ±25% {bb['amp_pct']:.0f}%",
                bb["n"],
            )
            c3 = (
                _cell(
                    "measured",
                    f"rhythm agrees {pd['rhythm_agrees_pct']:.0f}% vs native",
                    pd["n_with_native_rhythm"],
                )
                if pd
                else _cell("untested", "not simulated", 0)
            )
        return {"condition": label, "cells": [c1, c2, c3, ET]}

    rows.append(rate_row("Simulated phone 60 Hz", 60.0))
    rows.append(rate_row("Simulated 30 Hz", 30.0))
    rows.append(rate_row("Simulated 20 Hz", 20.0))

    e4 = X["E4_speed_stress"]["heldout"]
    inj = [e4["x2"], e4["x3"]]
    n_inj = sum(v["n_injected"] for v in inj)
    abst = sum(v["speed_gate_abstains_pct"] * v["n_injected"] / 100 for v in inj) / max(n_inj, 1) * 100
    kept_n = sum(v["recovery_kept"].get("n", 0) for v in inj)
    kept_ok = sum(
        v["recovery_kept"].get("freq_within_0p5_pct_reported", 0) * v["recovery_kept"].get("n", 0) / 100
        for v in inj
    )
    kept_pct = 100 * kept_ok / kept_n if kept_n else float("nan")
    n_null = sum(v["n_null"] for v in inj)
    n_null_kept = sum(v["n_null_kept"] for v in inj)
    null_abst = 100.0 * (1 - n_null_kept / n_null) if n_null else float("nan")
    fr_kept = (
        sum((v["false_rhythm_pct_kept"] or 0) * v["n_null_kept"] / 100 for v in inj) / n_null_kept * 100
        if n_null_kept
        else float("nan")
    )
    rows.append(
        {
            "condition": "Fast drawing ×2/×3 (time-compressed)",
            "cells": [
                _cell(
                    "abstains" if null_abst >= 50 else ("measured" if fr_kept <= 5 else "fails"),
                    f"abstains {null_abst:.0f}%; false rhythm {fr_kept:.0f}% kept",
                    n_null,
                ),
                _cell(
                    "abstains" if abst >= 50 else ("measured" if kept_pct >= 90 else "fails"),
                    f"speed gate abstains {abst:.0f}%; kept {kept_pct:.0f}%"
                    if kept_n
                    else f"speed gate abstains {abst:.0f}% (all)",
                    n_inj,
                ),
                _cell("untested", "PD drawings not compressed", 0),
                ET,
            ],
        }
    )
    for label in ("Finger on real phone", "Stylus on real phone"):
        rows.append(
            {
                "condition": label,
                "cells": [
                    _cell("untested", "no phone recordings yet", 0),
                    _cell("untested", "no phone recordings yet", 0),
                    _cell("untested", "no phone recordings yet", 0),
                    ET,
                ],
            }
        )
    return {"columns": cols, "rows": rows}


def headline(R):
    X = R["experiments"]
    E1h = X["E1_injection_recovery"]["heldout"]
    b = E1h["bars_pp_ge_1mm"]
    e2 = X["E2_specificity"]["heldout"]
    e3 = X["E3_phone_sampling"]["60"]["heldout"]["bars_pp_ge_1mm"]
    e5 = X["E5_baselines"]["linear"]
    e6 = X["E6_pd_vs_control_exploratory"]["heldout"]
    e7 = X["E7_static_vs_dynamic"]["heldout"]
    out = []
    geo = ", ".join(f"{g} {b[g]['freq_pct']:.0f}%" for g in GEOMS)
    out.append(
        {
            "claim": f"Injected 4–12 Hz tremor ≥1 mm pp on real control drawings: frequency reported within "
            f"±0.5 Hz in {geo}.",
            "n": sum(b[g]["n"] for g in GEOMS),
            "split": "held-out",
            "caveat": "synthetic sinusoidal tremor on real tablet drawings; not patient tremor",
        }
    )
    amp = ", ".join(f"{g} {b[g]['amp_pct']:.0f}%" for g in GEOMS)
    out.append(
        {
            "claim": f"Amplitude within ±25% of truth (pp ≥1 mm): {amp}.",
            "n": sum(b[g]["n"] for g in GEOMS),
            "split": "held-out",
            "caveat": "mm values assume 0.204 mm/px tablet scale",
        }
    )
    out.append(
        {
            "claim": f"False 'dominant rhythm' on un-injected healthy drawings: {e2['false_rhythm_n_new']}/"
            f"{e2['n_drawings']} (prominence ≥ {X['E2_specificity']['MIN_PROMINENCE']:g} and amplitude ≥ "
            f"{X['E2_specificity']['MIN_RHYTHM_PP_MM']:g} mm).",
            "n": e2["n_drawings"],
            "split": "held-out",
            "caveat": "8 control subjects; CI is wide",
        }
    )
    out.append(
        {
            "claim": f"Simulated 60 Hz phone sampling: ±0.5 Hz in {e3['freq_pct']:.0f}%, amplitude ±25% in "
            f"{e3['amp_pct']:.0f}% (pp ≥1 mm).",
            "n": e3["n"],
            "split": "held-out",
            "caveat": "simulated phone sampling, not a phone; finger/stylus on a real phone untested",
        }
    )
    out.append(
        {
            "claim": f"Linear-axis tremor vs a radial-only (Longardner-style) baseline: frequency ±0.5 Hz "
            f"{e5['freq_within_0p5_pct']['ours_2d']:.0f}% vs {e5['freq_within_0p5_pct']['radial']:.0f}%; "
            f"amplitude ratio {e5['amp_ratio_median']['ours_2d']:.2f} vs "
            f"{e5['amp_ratio_median']['radial']:.2f}.",
            "n": e5["n_trials"],
            "split": "held-out",
            "caveat": "advantage is amplitude, not frequency; same synthetic injections",
        }
    )
    a = e6["amp_pp_mm (log)"]
    out.append(
        {
            "claim": f"PD vs control, tremor amplitude AUC {a['auc']:.2f} (95% CI {a['ci95'][0]:.2f}–"
            f"{a['ci95'][1]:.2f}).",
            "n": e6["n_control"] + e6["n_pd"],
            "split": "held-out",
            "caveat": f"exploratory, {e6['n_control']} controls; device clock differs by group; not diagnostic",
        }
    )
    out.append(
        {
            "claim": f"Static vs dynamic spiral amplitude ICC(2,1) {e7['icc_2_1']:.2f}; MDC95 "
            f"{e7['mdc95_pct_change']:.0f}% change.",
            "n": e7["n_subjects"],
            "split": "held-out",
            "caveat": "between-task, not test-retest; Elble 2017 reports MDC 51% test-retest in ET",
        }
    )
    out.append(
        {
            "claim": "Essential tremor, real phones: untested.",
            "n": 0,
            "split": "n/a",
            "caveat": "UCI has no ET patients and no phone recordings",
        }
    )
    return out


# ------------------------------------------------------------------ EVAL.md


def _f(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def render_md(R):
    X = R["experiments"]
    T = R["thresholds"]
    L = []
    w = L.append
    w("# Spiral Test — evaluation\n")
    w(
        "Generated by `uv run python eval/run_eval.py` (seed "
        f"{R['config']['seed']}, {R['meta']['n_trials']} pipeline runs, {R['meta']['runtime_s']} s). "
        "Do not edit by hand; rerun instead.\n"
    )
    w(
        f"Data: {R['meta']['dataset']}. Every mm value assumes {R['config']['mm_per_px_assumed']} mm/px "
        "(no calibration in the dataset); frequencies and AUCs do not depend on it.\n"
    )
    if R["config"]["quick"]:
        w("**QUICK RUN — tiny grid, not for reporting.**\n")
    w("## Headline (held-out split)\n")
    for h in R["headline"]:
        w(f"- {h['claim']} *n={h['n']}, {h['split']}; caveat: {h['caveat']}.*")
    w("")

    w("## E0 Pre-registered split\n")
    sp = R["config"]["split"]
    for g in ("control", "parkinson"):
        w(f"- **{g}** dev ({len(sp[g]['dev'])}): {', '.join(sp[g]['dev'])}")
        w(f"- **{g}** held-out ({len(sp[g]['heldout'])}): {', '.join(sp[g]['heldout'])}")
    w(
        "\nSplit is by subject (both spiral tests of a subject stay together), seeded permutation of sorted IDs. "
        "Thresholds below were chosen on dev only. Whether `new_dataset` PD subjects overlap `hw_dataset` PD subjects "
        "is not documented; they are treated as distinct people.\n"
    )

    w("## Threshold changes (set on dev)\n")
    w("| constant | old | new | rule / evidence |")
    w("|---|---|---|---|")
    t1 = T["MIN_PROMINENCE"]
    dp = t1["dev_control_prominences"]
    w(
        f"| `MIN_PROMINENCE` | {t1['old']} | {t1['new']} | {t1['rule']}. Dev control prominences (n={len(dp)}): "
        f"max {_f(max(dp), 2)}, values ≥4: {[round(v, 2) for v in dp if v >= 4]} |"
    )
    t2 = T["GATE.min_rate_hz"]
    ev = "; ".join(
        f"{k} Hz: freq {_f(v.get('freq_pct'), 0)}%, amp {_f(v.get('amp_pct'), 0)}%"
        for k, v in t2["dev_bars_by_rate"].items()
    )
    w(
        f"| `GATE['min_rate_hz']` | {t2['old']} | {t2['new']} | {t2['rule']}. Validated rate "
        f"{t2['validated_rate_hz']} Hz. Dev bars (pp ≥1 mm): {ev} |"
    )
    ta = T["MIN_RHYTHM_PP_MM"]
    w(
        f"| `MIN_RHYTHM_PP_MM` (new constant) | {ta['old']} | {ta['new']} | {ta['rule']}. Largest dev control "
        f"amplitude {_f(ta['dev_control_max_amp_mm'], 3)} mm. A dominant rhythm now needs prominence AND amplitude. |"
    )
    t3 = T["GATE.max_speed_mm_s"]
    crv = t3["dev_curve"]
    tail = "; ".join(f"≤{c['max_speed']:g}: {_f(c['false_rhythm_pct'], 1)}% (n={c['n']})" for c in crv[-4:])
    w(
        f"| `GATE['max_speed_mm_s']` | {t3['old']} (first run of this eval: {t3['first_run_value']}) | {t3['new']} | "
        f"{t3['rule']}. Dev cumulative false rhythm, last steps: {tail}. Without the other-gates condition the "
        f"rule gives {t3['unconditional_value']}. The first run's rule ({t3['first_run_rule']}) is superseded. |"
    )
    w(
        f"| `GATE['max_template_dev_mm']` | {T['GATE.max_template_dev_mm']['old']} | "
        f"{T['GATE.max_template_dev_mm']['new']} | {T['GATE.max_template_dev_mm']['rule']}. See E10. |"
    )
    w(
        "| `GATE['max_clock_backsteps']` (new) | — | 0 | Any sample stamped >0.5 ms before an earlier one means the "
        "clock jumped; `clean()` no longer sorts (see the 11 Hz finding). |"
    )
    w(
        f"\n`spiral/metrics.py` currently holds MIN_PROMINENCE={t1['in_metrics_py']}, MIN_RHYTHM_PP_MM="
        f"{ta['in_metrics_py']}, min_rate_hz="
        f"{t2['in_metrics_py']}, max_speed_mm_s={t3['in_metrics_py']} → "
        f"{'matches' if R['thresholds_match_metrics_py'] else '**does not match**'} the dev-derived values. "
        "All held-out numbers in this file use the dev-derived values.\n"
    )

    E10 = X["E10_template_gate_synthetic"]
    w("## E10 Template gate (synthetic traces only — untested on real phone drawings)\n")
    w(
        "`template_deviation` used to measure distance to the *nearest* arm, which is capped at half the arm pitch "
        "(≈4.3 mm on the phone template) while the threshold was 10 mm, so `off_template` could never fire. It now "
        "follows the trace's own unwrapped angle: the first point fixes the start (any arm), and the expected radius "
        "grows by b per radian traced; it is computed on the low-passed path so tremor does not count. "
        f"{E10['label']}\n"
    )
    w(
        "| split | trace kind | n | median dev mm | p99 dev mm | off_template % (new) | any template gate % | off_template % (old nearest-arm, 10 mm) |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for s in ("heldout", "dev"):
        for k, v in E10[s].items():
            w(
                f"| {s} | {k} | {v['n']} | {_f(v['template_dev_median_mm'], 2)} | {_f(v['template_dev_p99_mm'], 2)} | "
                f"{_f(v['off_template_pct'])} | {_f(v['any_template_gate_pct'])} | {_f(v['old_nearest_arm_off_template_pct'])} |"
            )
    w("")

    w("## E1 Injection recovery (real control drawings + synthetic tremor)\n")
    w(
        "Tremor added to the raw, irregularly-sampled tablet samples, then `compute()` (ungated). "
        "'Reported' = prominence ≥ MIN_PROMINENCE **and** |f̂−f| ≤ 0.5 Hz (what the product would show); "
        "'any prom.' ignores the prominence threshold.\n"
    )
    for s in ("heldout", "dev"):
        E = X["E1_injection_recovery"][s]
        w(f"### {s} ({E['n_drawings']} drawings)\n")
        w(
            "| geometry | pp mm | n | reported ±0.5 Hz % | any prom. ±0.5 Hz % | median abs f err Hz | detect % | "
            "amp ratio median [IQR] | amp ±25% % |"
        )
        w("|---|---|---|---|---|---|---|---|---|")
        for g in GEOMS:
            for k, v in E["by_geometry_amplitude"][g].items():
                w(
                    f"| {g} | {k[:-2]} | {v['n']} | {_f(v['freq_within_0p5_pct_reported'])} | "
                    f"{_f(v['freq_within_0p5_pct_any_prominence'])} | {_f(v['median_abs_freq_err_hz'], 3)} | "
                    f"{_f(v['detection_pct'])} | {_f(v['amp_ratio_median'], 2)} [{_f(v['amp_ratio_iqr'][0], 2)}–"
                    f"{_f(v['amp_ratio_iqr'][1], 2)}] | {_f(v['amp_within_25pct_pct'])} |"
                )
        w("\n**Pre-registered bars (pp ≥ 1 mm, 4–12 Hz; ≥90% of trials):**\n")
        w("| geometry | n | freq reported ±0.5 Hz % | pass | amp within ±25% % | pass |")
        w("|---|---|---|---|---|---|")
        for g, bb in E["bars_pp_ge_1mm"].items():
            w(
                f"| {g} | {bb['n']} | {_f(bb['freq_pct'])} | {'PASS' if bb['freq_pass'] else 'FAIL'} | "
                f"{_f(bb['amp_pct'])} | {'PASS' if bb['amp_pass'] else 'FAIL'} |"
            )
        w(
            "\nBy injected frequency (pp ≥ 1 mm, all geometries): "
            + ", ".join(
                f"{k} Hz {_f(v['freq_within_0p5_pct_reported'], 0)}%/{_f(v['amp_within_25pct_pct'], 0)}%"
                for k, v in E["by_frequency_pp_ge_1mm"].items()
            )
            + " (reported-freq % / amp ±25% %)\n"
        )
    w("![E1](fig/e1_recovery.png)\n")

    E2 = X["E2_specificity"]
    w("## E2 Specificity (un-injected healthy drawings)\n")
    w(
        f"False rhythm = prominence ≥ {E2['MIN_PROMINENCE']} AND amp ≥ {E2['MIN_RHYTHM_PP_MM']} mm (new rule).\n"
    )
    w(
        "| split | drawings | false rhythm @new | prominence-only @new | @old 4.0 | max prominence | freqs of false rhythms | "
        "false rhythm @new, simulated 60 Hz |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for s in ("heldout", "dev"):
        e = E2[s]
        w(
            f"| {s} | {e['n_drawings']} | {e['false_rhythm_n_new']} ({_f(e['false_rhythm_pct_new'])}%) | "
            f"{e['false_rhythm_n_prominence_only']} | {_f(e['false_rhythm_pct_old'])}% | {_f(e['max_prominence'], 2)} | {e['false_rhythm_freqs_hz']} | "
            f"{_f(e['false_rhythm_pct_sim60'])}% (n={e['n_sim60']}) |"
        )
    w(
        "\nDetection rate on injected tremor (held-out, all geometries), old prominence 4.0 → prominence-only "
        "new → new prominence AND amplitude: "
        + "; ".join(
            f"{k}: {_f(v['detect_pct_old'], 0)}% → {_f(v['detect_pct_prominence_only'], 0)}% → {_f(v['detect_pct_new'], 0)}%"
            for k, v in E2["heldout"]["detection_cost_by_amplitude"].items()
        )
        + "\n"
    )
    w("![prominence](fig/prominence.png)\n")

    E3 = X["E3_phone_sampling"]
    w("## E3 Simulated phone sampling (not a phone)\n")
    w(
        f"{E3['label']}. iOS Safari caps pointer events at 60/s even on 120 Hz screens "
        "(https://bugs.webkit.org/show_bug.cgi?id=261961). Gate min_rate = "
        f"{E3['min_rate_hz']} Hz.\n"
    )
    w(
        "| rate Hz | split | n | reported ±0.5 Hz % | amp ±25% % | bars | 4–6 / 7–9 / 10–12 Hz % | "
        "amp ratio median | gate abstains % | false rhythm (un-injected) % |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for rate in RATES:
        for s in ("heldout", "dev"):
            d = E3[str(int(rate))][s]
            bb = d["bars_pp_ge_1mm"]
            fb = d["freq_pct_by_band"]
            w(
                f"| {rate:g} | {s} | {bb['n']} | {_f(bb['freq_pct'])} | {_f(bb['amp_pct'])} | "
                f"{'PASS' if bb['freq_pass'] and bb['amp_pass'] else 'FAIL'} | {_f(fb['4-6'], 0)} / {_f(fb['7-9'], 0)} / "
                f"{_f(fb['10-12'], 0)} | {_f(d['recovery_pp_ge_1mm']['amp_ratio_median'], 2)} | "
                f"{_f(d['gate_abstains_low_rate_pct'], 0)} | {_f(d['false_rhythm_pct_uninjected'], 0)} |"
            )
    w(
        "\nHeld-out per-geometry at 60 Hz: "
        + "; ".join(
            f"{g} freq {_f(v['freq_pct'], 0)}% / amp {_f(v['amp_pct'], 0)}%"
            for g, v in E3["60"]["heldout"]["by_geometry"].items()
        )
    )
    w(
        "\nParkinson's drawings degraded (all PD, un-injected; agreement = native rhythm reproduced within ±0.5 Hz): "
        + "; ".join(
            f"{k[3:]} Hz: agrees {_f(E3[k]['rhythm_agrees_pct'], 0)}% of {E3[k]['n_with_native_rhythm']}, "
            f"gate abstains {_f(E3[k]['gate_abstains_low_rate_pct'], 0)}%, median log amp ratio "
            f"{_f(E3[k]['amp_log_ratio_median'], 2)}"
            for k in ("pd_60", "pd_30", "pd_20")
        )
        + "\n"
    )
    w("![E3](fig/e3_sampling_rate.png)\n")

    E4 = X["E4_speed_stress"]
    w("## E4 Speed stress (time-compressed real drawings, injected 5 Hz, 1 mm pp)\n")
    w(f"Gate max_speed = {E4['max_speed_mm_s']} mm/s. Confound: {E4['confound']}.\n")
    hk = E4["heldout_gate_kept_pooled"]
    w(
        f"Held-out check of the full gate (all factors pooled): injected trials kept {hk['n_injected_kept']}, reported "
        f"±0.5 Hz {_f(hk['recovery']['freq_within_0p5_pct_reported'])}%, amp ±25% "
        f"{_f(hk['recovery']['amp_within_25pct_pct'])}%; un-injected kept {hk['n_uninjected_kept']} with false rhythm "
        f"{_f(hk['false_rhythm_pct_kept'])}%, versus {_f(hk['false_rhythm_pct_rejected_too_fast'])}% in the "
        f"{hk['n_uninjected_rejected_too_fast']} un-injected drawings the speed gate rejects.\n"
    )
    w(
        "| split | factor | n | median speed mm/s | reported ±0.5 Hz % (all) | amp ±25% % (all) | false rhythm % (all) | "
        "passes other gates % | speed gate abstains % | reported % (kept) |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("heldout", "dev"):
        for k, v in E4[s].items():
            w(
                f"| {s} | {k} | {v['n_injected']} | {_f(v['speed_median_mm_s'], 0)} | "
                f"{_f(v['recovery_all']['freq_within_0p5_pct_reported'])} | "
                f"{_f(v['recovery_all']['amp_within_25pct_pct'])} | {_f(v['false_rhythm_pct_all'], 0)} | "
                f"{_f(v['passes_other_gates_pct'], 0)} | {_f(v['speed_gate_abstains_pct'], 0)} | {_f(v['recovery_kept'].get('freq_within_0p5_pct_reported'))} |"
            )
    w("")

    E5 = X["E5_baselines"]
    w("## E5 Baselines (same held-out injections, pp ≥ 1 mm)\n")
    w(
        f"Radial: {E5['method_radial']}. Zero-cross: {E5['method_zero_cross']}. Frequency success here ignores "
        "prominence for all three methods.\n"
    )
    w(
        "| geometry | n | ours 2-D % | radial % | zero-cross % | ours−radial pp [95% CI] | ours−zero-cross pp [95% CI] | "
        "amp ratio ours / radial | median paired abs-log-amp err radial−ours [CI] |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for g in [*GEOMS, "all"]:
        e = E5[g]
        fr = e["freq_within_0p5_pct"]
        dr = e["paired_diff_ours_minus_radial_pct_points"]
        dz = e["paired_diff_ours_minus_zero_cross_pct_points"]
        da = e["median_paired_abs_log_amp_err_radial_minus_ours"]
        w(
            f"| {g} | {e['n_trials']} | {_f(fr['ours_2d'])} | {_f(fr['radial'])} | {_f(fr['zero_cross'])} | "
            f"{_f(dr['est'])} [{_f(dr['ci95'][0])}, {_f(dr['ci95'][1])}] | {_f(dz['est'])} [{_f(dz['ci95'][0])}, "
            f"{_f(dz['ci95'][1])}] | {_f(e['amp_ratio_median']['ours_2d'], 2)} / {_f(e['amp_ratio_median']['radial'], 2)}"
            f" | {_f(da['est'], 3)} [{_f(da['ci95'][0], 3)}, {_f(da['ci95'][1], 3)}] |"
        )
    w("")

    E6 = X["E6_pd_vs_control_exploratory"]
    w("## E6 PD vs control on real drawings — EXPLORATORY, not diagnostic\n")
    w(
        f"{E6['label']}. AUC = P(PD subject > control subject); 1000× seeded bootstrap CI over subjects within group. "
        f"Excluded drawings: {E6['excluded_drawings']}.\n"
    )
    feats = [k for k in E6["heldout"] if k not in ("n_control", "n_pd")]
    w("| cohort | controls | PD | " + " | ".join(feats) + " |")
    w("|---|---|---|" + "---|" * len(feats))
    for c in ("heldout", "dev", "all_subjects", "all_controls_vs_new_dataset_PD_only (same ~142 Hz clock)"):
        e = E6[c]
        w(
            f"| {c} | {e['n_control']} | {e['n_pd']} | "
            + " | ".join(
                f"{_f(e[k]['auc'], 2)} [{_f(e[k]['ci95'][0], 2)}–{_f(e[k]['ci95'][1], 2)}]" for k in feats
            )
            + " |"
        )
    p = E6["pd_dominant_rhythm"]
    w(
        f"\nDominant rhythm (prominence ≥ {T['MIN_PROMINENCE']['new']}): {p['n_with_rhythm']}/{p['n_pd_drawings']} PD "
        f"drawings ({_f(p['pct_with_rhythm'], 0)}%) vs {p['control_drawings_with_rhythm']}/{p['n_control_drawings']} "
        f"control drawings. PD rhythm frequency median {_f(p['freq_median_hz'], 2)} Hz, IQR "
        f"{_f(p['freq_iqr_hz'][0], 2)}–{_f(p['freq_iqr_hz'][1], 2)} Hz; histogram {p['freq_histogram']}. "
        f"{p['n_with_rhythm_and_gate_pass']} of the rhythmic PD drawings also pass the gate "
        f"({p['n_gate_pass']} PD drawings pass the gate in total).\n"
    )
    w(
        "Sampling-clock confound: "
        + "; ".join(
            f"{k}: {_f(v['median_nominal_rate_hz'], 0)} Hz (n={v['n_drawings']})"
            for k, v in E6["sampling_rate_confound"].items()
        )
        + ". The hw_dataset PD drawings were recorded on a ~111 Hz clock and the controls on ~143 Hz, so any group "
        "difference can partly be a device difference; the new_dataset-only row is the same-clock check.\n"
    )

    E7 = X["E7_static_vs_dynamic"]
    w("## E7 Agreement: static vs dynamic spiral (between-task, not test-retest)\n")
    w(
        f"{E7['label']}. Reference: {E7['reference']}. ICC(2,1) on natural-log pp amplitude, subject-bootstrap CI; "
        "SEM = √(Σd²/2n) (includes the systematic task difference); MDC95 = 1.96·√2·SEM, shown as % change.\n"
    )
    w("| cohort | subjects | ICC(2,1) [95% CI] | SEM (log) | MDC95 % change | mean log(dyn/static) |")
    w("|---|---|---|---|---|---|")
    for c in ("heldout", "all_subjects", "parkinson_all", "control_all"):
        e = E7[c]
        w(
            f"| {c} | {e['n_subjects']} | {_f(e['icc_2_1'], 2)} [{_f(e['icc_ci95'][0], 2)}–{_f(e['icc_ci95'][1], 2)}] | "
            f"{_f(e['sem_log'], 3)} | {_f(e['mdc95_pct_change'], 0)}% | {_f(e['mean_log_diff_dynamic_minus_static'], 3)} |"
        )
    w("\n![E7](fig/e7_agreement.png)\n")

    E8 = X["E8_gate_real_data"]
    w("## E8 Gate on real drawings\n")
    w(f"{E8['note']}.\n")
    w("| group/split | drawings | abstain % | reasons |")
    w("|---|---|---|---|")
    for k, v in E8.items():
        if "/" in k:
            w(
                f"| {k} | {v['n_drawings']} | {_f(v['abstain_pct'], 0)} ({v['n_abstain']}) | {v['reason_counts']} |"
            )
    w(f"\nCorrupted (clock-reversal) drawings and their gate result: {E8['corrupted_drawings_gate']}\n")

    A = X["audit_11hz"]
    w("## Finding: the 11 Hz 'tremor' is a timestamp artefact\n")
    w(
        "Two PD drawings showed an ~11 Hz peak with very high prominence but tiny pp amplitude. Both are the only "
        "drawings whose tablet clock jumps **backwards** mid-recording. The old `clean()` sorted by time, so the block "
        "after the jump was interleaved sample-by-sample with the block before it; the trace then alternated between "
        "two places every sample. That toggle sits at the tablet's Nyquist rate (≈55.6 Hz for a 9 ms clock), and linear "
        "interpolation onto the 100 Hz grid folds it to |111.1 − 100| = 11.1 Hz. (The tablet clock is 7/8 ms or 9/10 ms, "
        "not alternating 7/9 ms; the 7–8 ms files fold to ≈43 Hz, outside the band, which is why only ~111 Hz files show "
        "it.) **Fixed:** `clean()` now keeps file order and drops samples that go back in time; `compute()` counts them "
        "(`clock_backsteps`) and the gate abstains with reason `clock`.\n"
    )
    w(
        "| drawing | clock | jump ms | samples behind the clock | old sorting clean(): f / prom / amp pp / gate | "
        "file order repaired: f / prom / amp pp | now: backsteps / gate |"
    )
    w("|---|---|---|---|---|---|---|")
    for d in A["drawings"]:
        o, a, b = d["old_sorting_clean"], d["as_recorded"], d["file_order_repaired"]
        w(
            f"| {d['id']} | {_f(d['nominal_rate_hz'], 0)} Hz | {_f(d['clock_jump_ms'], 0)} | "
            f"{d['samples_interleaved_after_sort']} | {_f(o['freq_hz'], 2)} / {_f(o['prominence'], 1)} / "
            f"{_f(o['amp_pp_mm'], 3)} / {o['gate'] or 'passes'} | {_f(b['freq_hz'], 2)} / {_f(b['prominence'], 1)} / "
            f"{_f(b['amp_pp_mm'], 3)} | {a['clock_backsteps']} / {a['gate'] or 'passes'} |"
        )
    w(
        "\nSynthetic reproduction of the mechanism (smooth spiral, every other sample shifted 1 mm): "
        + "; ".join(
            f"{_f(s['sample_rate_hz'], 1)} Hz clock → peak {_f(s['freq_hz'], 2)} Hz, prominence {_f(s['prominence'], 1)} "
            f"(predicted alias {_f(s['predicted_alias_hz'], 1)} Hz)"
            for s in A["synthetic"]
        )
        + ".\n"
    )
    w("These two drawings remain excluded from E6/E7 (their timing is unrecoverable).\n")

    E9 = X["E9_mutations"]
    w("## E9 Mutation checks (monkeypatched in-process; metrics.py untouched)\n")
    cn = list(E9["table"]["none"].keys())
    w("| mutation | " + " | ".join(cn) + " | expected red | caught |")
    w("|---|" + "---|" * len(cn) + "---|---|")
    for m, row in E9["table"].items():
        cells = " | ".join(("🟢" if row[c]["green"] else "🔴") + " " + row[c]["detail"] for c in cn)
        v = E9["verdict"].get(m)
        w(
            f"| {m} | {cells} | {v['expected_red'] if v else '—'} | "
            f"{('yes' if v['caught'] else '**NO — finding**') if v else ('all green' if E9['unmutated_all_green'] else '**baseline not green**')} |"
        )
    w(
        "\nThe last column was added after the first full run, because the pre-defined `specificity` check (false rhythm on "
        "gate-passing drawings, ≤10%) did not turn red when the local-maximum / band-edge rule was removed: on the real "
        "control drawings the gate admits, the rule changes very few decisions. "
        + (
            "The post-hoc check (all un-injected held-out controls, no gate) does turn red. "
            if not E9["table"]["no_band_edge_rejection"]["specificity_ungated (added post hoc)"]["green"]
            else "With MIN_RHYTHM_PP_MM in place, the post-hoc check (all un-injected held-out controls, no gate) "
            "stays green too: the amplitude floor already removes the edge-leakage peaks it would have caught. "
        )
        + "No check in this harness shows the local-maximum rule doing protective work on real drawings; treat "
        "its effect as unproven (it is still the principled rule for a 4.0 Hz peak).\n"
    )

    w("## Failure map\n")
    fm = R["failure_map"]
    w("| condition | " + " | ".join(fm["columns"]) + " |")
    w("|---|" + "---|" * len(fm["columns"]))
    for row in fm["rows"]:
        w(
            f"| {row['condition']} | "
            + " | ".join(f"**{c['status']}**: {c['text']} (n={c['n']})" for c in row["cells"])
            + " |"
        )
    w("")

    w("## Data issues found\n")
    w(
        "- UCI has no pen-up flag: pressure is never 0 and Z is not a contact flag. Pressure falls to the bottom "
        "decile immediately before ~98% of gaps > 50 ms, so gaps > 50 ms are treated as pen lifts (stroke index)."
    )
    w(
        "- `new_dataset/parkinson/H_P000-0024` and `H_P000-0025` contain a byte-identical dynamic-spiral recording; "
        "both copies are excluded from E6/E7."
    )
    w("- Two drawings have a backwards clock jump (see the 11 Hz finding).")
    w(
        "- `new_dataset/parkinson/H_p000-0017` contains only the circular-motion test (no spiral), so 61 of the 62 PD "
        "subjects contribute spirals."
    )
    w("- Control drawings were recorded at ~143 Hz, hw_dataset PD at ~111 Hz (see E6 confound).\n")

    w("## Issues found and fixed in this pass (with the orchestrator's permission to change metrics.py)\n")
    w(
        "1. **Clock jumps.** `clean()` sorted by time, interleaving two parts of a trace whose clock jumped back; "
        "`P_05060004/t0` passed the gate with an 11 Hz 'rhythm'. Now: file order kept, backward samples dropped and "
        "counted (`sampling.clock_backsteps`), gate reason `clock`. Repro (now abstains): `n=3000; t=np.arange(n)*9.; "
        "th=np.linspace(0,6*np.pi,n); x=(5+3*th)*np.cos(th); y=(5+3*th)*np.sin(th); t[1500:]-=1300; analyze(t,x,y)`."
    )
    w(
        "2. **Parabolic interpolation** ran even when the maximum was not a vertex (`spectral_peak(f, f**2)` returned "
        "0.0 Hz; `H_P000-0001/t1` gave −54 Hz). Now skipped unless the triple is concave, and clamped to ±0.5 bin."
    )
    w(
        "3. **Speed across lifts.** The straight-line bridge `resample()` draws across a pen lift counted as drawn path. "
        "Grid steps whose ends fall in different strokes, or inside an inter-stroke gap, are now excluded."
    )
    w(
        "4. **Template gate could never fire** (nearest-arm distance ≤ half the pitch < 10 mm threshold). Replaced by a "
        "trace-following deviation; threshold from synthetic traces only (E10)."
    )
    w("\n## Open issues\n")
    w(
        "1. `max_gap_ms`, `max_lifts`, `min_duration_s`, `min_coverage` were not re-tuned: no experiment here gives evidence for them."
    )
    w(
        "2. `max_template_dev_mm` and `MIN_RHYTHM_PP_MM` rest on assumptions: the honest-trace error model (E10) and "
        "the 0.204 mm/px tablet scale (MIN_RHYTHM_PP_MM). Both need real phone drawings."
    )
    w(
        "3. A trace that wanders across the spiral centre loses turns in the unwrapped angle (coverage); E10's "
        "synthetic traces never cross the centre, so this is untested.\n"
    )

    w("## Reproduce\n")
    w(
        "```\nuv sync\nuv run python scripts/fetch_uci.py   # if data/raw/uci is missing\nuv run python eval/run_eval.py\n```\n"
    )
    w(
        f"metrics.py sha1 at run time: `{R['meta']['metrics_py_sha1']}`. Workers: {R['meta']['workers']} (results do not "
        "depend on the worker count: every trial is seeded from its own spec).\n"
    )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
