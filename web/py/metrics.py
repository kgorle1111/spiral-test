"""Tremor metrics from one traced spiral.

Constraint: numpy + stdlib only. This exact file is loaded by the web app under
Pyodide, so the phone and the evaluation run the same code. Do not import scipy
or anything from this package here.

Units in, units out: time in ms, position in mm. Frequencies in Hz.
"""

from __future__ import annotations

import math

import numpy as np

FS = 100.0  # Hz, uniform grid every drawing is resampled to
BAND = (4.0, 12.0)  # tremor search band: ET 4-12 Hz, PD 4-6 Hz
HP = (2.5, 3.5)  # cosine ramp: below 2.5 Hz removed, above 3.5 Hz kept
LP = (12.0, 14.0)  # cosine ramp: above 14 Hz removed
TRIM_S = 0.5  # edge seconds dropped after filtering
SEG_S = 4.0  # Welch segment length -> 0.25 Hz bins
AMP_WIN_S = 0.5  # window for the principal-axis amplitude
AMP_HOP_S = 0.25

# Gate thresholds. Set on the development split of the evaluation
# (eval/run_eval.py); the held-out split never touched them.
GATE = {
    "min_rate_hz": 42.0,  # distinct samples per second
    "max_gap_ms": 250.0,  # longest silence inside a stroke
    "min_duration_s": 8.0,
    "max_lifts": 1,
    "min_coverage": 0.85,  # share of the template's angle actually traced
    "max_speed_mm_s": 40.0,
    "max_template_dev_mm": 1.8,  # RMS radial error vs the arm being traced; synthetic-only threshold
    "max_clock_backsteps": 0,  # samples timestamped before an earlier sample
}
MIN_PROMINENCE = 9.5  # peak PSD / median in-band PSD to call a "dominant rhythm"
MIN_RHYTHM_PP_MM = 0.45  # a "dominant rhythm" also needs this peak-to-peak amplitude
CLOCK_TOL_MS = 0.5  # backward steps smaller than this are timer noise, not a clock jump


# ---------------------------------------------------------------- input


def _finite(t_ms, x, y, stroke):
    t = np.asarray(t_ms, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.zeros_like(t) if stroke is None else np.asarray(stroke, dtype=float)
    if not (t.shape == x.shape == y.shape == s.shape):
        raise ValueError("t, x, y and stroke must have the same length")
    keep = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    return t[keep], x[keep], y[keep], s[keep]


def _prev_max(t):
    """Running maximum of the timestamps strictly before each sample (-inf for the first)."""
    return np.concatenate([[-np.inf], np.maximum.accumulate(t)[:-1]]) if t.size else t


def clock_backsteps(t_ms):
    """Samples stamped more than CLOCK_TOL_MS before an earlier sample (a clock jump)."""
    t = np.asarray(t_ms, dtype=float)
    t = t[np.isfinite(t)]
    return int(np.sum(t < _prev_max(t) - CLOCK_TOL_MS))


def clean(t_ms, x, y, stroke=None):
    """Keep file order, drop non-finite rows and samples that go back in time, collapse duplicates.

    Never sort: sorting a recording whose clock jumped back interleaves two parts of the
    trace sample by sample, which aliases into a fake ~11 Hz rhythm (see results/EVAL.md).
    """
    t, x, y, s = _finite(t_ms, x, y, stroke)
    fwd = t >= _prev_max(t)
    t, x, y, s = t[fwd], x[fwd], y[fwd], s[fwd]
    if t.size:
        # Coalesced pointer events can share a timestamp; keep the last one.
        last = np.append(t[1:] != t[:-1], True)
        t, x, y, s = t[last], x[last], y[last], s[last]
    return t, x, y, s.astype(int)


def sampling_stats(t_ms, stroke):
    """Effective rate and longest in-stroke gap, from the raw timestamps."""
    if t_ms.size < 2:
        return {"rate_hz": 0.0, "max_gap_ms": float("inf"), "duration_s": 0.0}
    dur = (t_ms[-1] - t_ms[0]) / 1000.0
    dt = np.diff(t_ms)
    same = stroke[1:] == stroke[:-1]
    in_stroke = dt[same]
    pen_time = float(in_stroke.sum()) / 1000.0
    n_in = int(same.sum())
    rate = n_in / pen_time if pen_time > 0 else 0.0
    return {
        "rate_hz": float(rate),
        "max_gap_ms": float(in_stroke.max()) if in_stroke.size else float("inf"),
        "duration_s": float(dur),
        "pen_time_s": float(pen_time),
    }


# ---------------------------------------------------------------- signal


def resample(t_ms, *sigs, fs=FS):
    """Linear interpolation onto a uniform grid. Returns (t_s, *sigs)."""
    t = (t_ms - t_ms[0]) / 1000.0
    n = int(math.floor(t[-1] * fs)) + 1
    grid = np.arange(n) / fs
    return (grid, *[np.interp(grid, t, s) for s in sigs])


def _ramp(f, lo, hi):
    """0 below lo, 1 above hi, raised cosine between."""
    r = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * r)


def bandpass(sig, fs=FS, hp=HP, lp=LP):
    """Zero-phase FFT band-pass with odd-reflection padding against edge ringing."""
    n = sig.size
    pad = min(n - 1, int(2 * fs))
    ext = np.pad(sig, pad, mode="reflect", reflect_type="odd")
    m = ext.size
    spec = np.fft.rfft(ext)
    f = np.fft.rfftfreq(m, 1.0 / fs)
    mask = _ramp(f, hp[0], hp[1]) * (1.0 - _ramp(f, lp[0], lp[1]))
    out = np.fft.irfft(spec * mask, n=m)
    return out[pad : pad + n]


def lowpass(sig, fs=FS, edge=HP):
    """The smooth drawing path: everything the band-pass removed below the band."""
    n = sig.size
    pad = min(n - 1, int(2 * fs))
    ext = np.pad(sig, pad, mode="reflect", reflect_type="odd")
    spec = np.fft.rfft(ext)
    f = np.fft.rfftfreq(ext.size, 1.0 / fs)
    out = np.fft.irfft(spec * (1.0 - _ramp(f, edge[0], edge[1])), n=ext.size)
    return out[pad : pad + n]


def welch(sig, fs=FS, seg_s=SEG_S):
    """One-sided PSD (mm^2/Hz), Hann window, 50% overlap."""
    nseg = int(round(seg_s * fs))
    if sig.size < nseg:
        nseg = sig.size
    hop = nseg // 2
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nseg) / nseg)
    scale = 1.0 / (fs * np.sum(w * w))
    acc = np.zeros(nseg // 2 + 1)
    k = 0
    for start in range(0, sig.size - nseg + 1, max(hop, 1)):
        seg = sig[start : start + nseg]
        seg = seg - seg.mean()
        acc += np.abs(np.fft.rfft(seg * w)) ** 2
        k += 1
    psd = acc * scale / max(k, 1)
    psd[1:-1] *= 2.0
    return np.fft.rfftfreq(nseg, 1.0 / fs), psd


def spectral_peak(freqs, psd, band=BAND):
    """Peak frequency with parabolic sub-bin interpolation, and its prominence."""
    inb = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    i = int(inb[np.argmax(psd[inb])])
    f_peak = float(freqs[i])
    # A rhythm is a local maximum. A band-edge maximum whose out-of-band
    # neighbour is higher is leakage from voluntary motion, not a rhythm.
    edge = not (0 < i < psd.size - 1 and psd[i] > psd[i - 1] and psd[i] >= psd[i + 1])
    if 0 < i < psd.size - 1:
        a, b, c = psd[i - 1], psd[i], psd[i + 1]
        den = a - 2 * b + c
        if den < 0:  # only a concave triple has a vertex worth interpolating to
            off = min(0.5, max(-0.5, float(0.5 * (a - c) / den)))
            f_peak += off * float(freqs[1] - freqs[0])
    others = psd[inb][np.abs(freqs[inb] - freqs[i]) > 1.0]
    floor = float(np.median(others)) if others.size else float("nan")
    prom = float(psd[i] / floor) if floor > 0 else float("inf")
    return f_peak, (0.0 if edge else prom)


def pp_amplitude(rx, ry, fs=FS, win_s=AMP_WIN_S, hop_s=AMP_HOP_S):
    """Peak-to-peak tremor amplitude (mm) along the local dominant axis.

    Per short window: 2 * sqrt(2 * largest covariance eigenvalue). For a sinusoid
    A*sin along any fixed axis this is 2A; for circular motion of radius A it is
    2A too. Short windows follow the axis as it turns with the spiral.
    Returns the median over windows.
    """
    w = int(round(win_s * fs))
    h = max(1, int(round(hop_s * fs)))
    vals = []
    for s in range(0, rx.size - w + 1, h):
        a, b = rx[s : s + w], ry[s : s + w]
        a = a - a.mean()
        b = b - b.mean()
        sxx, syy, sxy = np.mean(a * a), np.mean(b * b), np.mean(a * b)
        lam = 0.5 * (sxx + syy) + math.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
        vals.append(2.0 * math.sqrt(2.0 * lam))
    return float(np.median(vals)) if vals else float("nan")


def pen_down_segments(t_ms, stroke, grid_s):
    """For each step of the uniform grid, True if both ends lie inside one pen-down stroke.

    resample() bridges a lift with a straight line; those steps are not drawing and
    must not count towards speed.
    """
    t = (t_ms - t_ms[0]) / 1000.0
    j = np.clip(np.searchsorted(t, grid_s, side="right") - 1, 0, t.size - 1)
    nxt = np.minimum(j + 1, t.size - 1)
    inside_gap = stroke[j] != stroke[nxt]  # grid point falls between two strokes
    same = stroke[j[:-1]] == stroke[j[1:]]
    return same & ~inside_gap[:-1] & ~inside_gap[1:]


# ---------------------------------------------------------------- geometry


def unwrap_angle(x, y, cx, cy):
    return np.unwrap(np.arctan2(y - cy, x - cx))


def fit_center(x, y, span_mm=15.0, step_mm=1.0):
    """Centre that makes the trace most like r = a + b*theta (grid search, then refine)."""
    cx0, cy0 = float(np.mean(x)), float(np.mean(y))

    def cost(cx, cy):
        th = unwrap_angle(x, y, cx, cy)
        r = np.hypot(x - cx, y - cy)
        A = np.stack([np.ones_like(th), th], axis=1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        return float(np.mean((r - A @ coef) ** 2))

    best = (cost(cx0, cy0), cx0, cy0)
    for span, step in ((span_mm, step_mm * 3), (step_mm * 3, step_mm), (step_mm, step_mm / 4)):
        cx_c, cy_c = best[1], best[2]
        grid = np.arange(-span, span + 1e-9, step)
        for dx in grid:
            for dy in grid:
                c = cost(cx_c + dx, cy_c + dy)
                if c < best[0]:
                    best = (c, cx_c + dx, cy_c + dy)
    return best[1], best[2]


def template_deviation(x, y, tpl):
    """RMS radial distance (mm) from where the trace should be, following its own angle.

    tpl: {cx, cy, a, b, turns, phi0, dir}; the printed spiral is r = a + b*alpha with
    alpha = dir*(angle - phi0), arm k covering alpha in [2*pi*k, 2*pi*(k+1)).
    The trace's angle is unwrapped around the template centre. Its first point fixes
    where on an arm it starts; from there the expected radius changes by b per radian
    actually traced. The start arm k0 is whichever fits best, so a trace may begin at
    the centre or at the outer end. Distance to the *nearest* arm could never exceed
    half the arm pitch, so circles, scribbles or a trace between arms passed; here they
    accumulate error as their angle advances without the radius following.
    """
    cx, cy, a, b = tpl["cx"], tpl["cy"], tpl["a"], tpl["b"]
    alpha = tpl.get("dir", 1) * (np.unwrap(np.arctan2(y - cy, x - cx)) - tpl.get("phi0", 0.0))
    r = np.hypot(x - cx, y - cy)
    a0 = float(np.mod(alpha[0], 2 * np.pi))
    prog = alpha - alpha[0]
    best = float("inf")
    # k = -1 covers a start a hair before phi0 (a0 just under 2*pi) on the innermost arm.
    for k in range(-1, int(math.ceil(tpl["turns"])) + 1):
        d = r - (a + b * (a0 + 2 * np.pi * k + prog))
        best = min(best, float(math.sqrt(np.mean(d * d))))
    return best


def radial_residual(x, y, cx, cy):
    """Unwrap the spiral: distance from the best-fit r = a + b*theta, in mm."""
    th = unwrap_angle(x, y, cx, cy)
    r = np.hypot(x - cx, y - cy)
    A = np.stack([np.ones_like(th), th], axis=1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return th, r - A @ coef


# ---------------------------------------------------------------- main


def compute(t_ms, x_mm, y_mm, stroke=None, template=None):
    """Every number, with no gate applied. The evaluation calls this directly.

    template: optional dict {cx, cy, a, b, turns} in mm, r = a + b*theta.
    """
    backsteps = clock_backsteps(t_ms)
    t, x, y, s = clean(t_ms, x_mm, y_mm, stroke)
    if t.size < 10:
        raise ValueError("fewer than 10 samples")
    samp = sampling_stats(t, s)
    samp["clock_backsteps"] = backsteps
    tg, xg, yg = resample(t, x, y)
    rx, ry = bandpass(xg), bandpass(yg)
    trim = int(TRIM_S * FS)
    if rx.size > 2 * trim + int(FS):
        rx, ry, tg2 = rx[trim:-trim], ry[trim:-trim], tg[trim:-trim]
    else:
        tg2 = tg
    fx, px = welch(rx)
    _, py = welch(ry)
    psd = px + py  # direction-invariant: x and y power add
    f_peak, prom = spectral_peak(fx, psd)
    inband = (fx >= BAND[0]) & (fx <= BAND[1])
    band_power = float(np.sum(psd[inband]) * (fx[1] - fx[0]))
    sx, sy = lowpass(xg), lowpass(yg)
    seg_ok = pen_down_segments(t, s, tg)
    path = float(np.sum(np.hypot(np.diff(sx), np.diff(sy))[seg_ok]))
    speed = path / samp["pen_time_s"] if samp.get("pen_time_s") else float("nan")

    if template:
        cx, cy = template["cx"], template["cy"]
    else:
        cx, cy = fit_center(x, y)
    th, rad = radial_residual(xg, yg, cx, cy)
    out = {
        "sampling": samp,
        "n_samples": int(t.size),
        "lifts": int(np.unique(s).size - 1),
        "freq_hz": f_peak,
        "prominence": prom,
        "amp_pp_mm": pp_amplitude(rx, ry),
        "rms_mm": float(math.sqrt(np.mean(rx * rx + ry * ry))),
        "band_power_mm2": band_power,
        "speed_mm_s": speed,
        "path_mm": path,
        "center": [float(cx), float(cy)],
        "turns": float(abs(th[-1] - th[0]) / (2 * np.pi)),
        "series": {
            "t": tg2,
            "rx": rx,
            "ry": ry,
            "theta": th,
            "radial": rad,
            # The same distance with slow drawing drift removed: the tremor itself.
            "radial_bp": bandpass(rad),
            "freqs": fx,
            "psd": psd,
        },
    }
    if template:
        # On the smooth path: tremor is what we measure, not a failure to follow the line.
        out["template_dev_mm"] = template_deviation(sx, sy, template)
        out["coverage"] = float(min(1.0, abs(th[-1] - th[0]) / (2 * np.pi * template["turns"])))
    return out


def gate(m, gate_cfg=None):
    """Return the list of reasons to redo. Empty list means the drawing is usable."""
    g = dict(GATE)
    if gate_cfg:
        g.update(gate_cfg)
    reasons = []
    sp = m["sampling"]
    if sp.get("clock_backsteps", 0) > g.get("max_clock_backsteps", 0):
        reasons.append(("clock", "The device clock jumped; this recording can't be timed. Redo it."))
    if sp["rate_hz"] < g["min_rate_hz"]:
        reasons.append(
            (
                "low_rate",
                f"The screen sent {sp['rate_hz']:.0f} points per second; at least "
                f"{g['min_rate_hz']:.0f} are needed to see a 12 Hz tremor. Try a stylus or another browser.",
            )
        )
    if sp["max_gap_ms"] > g["max_gap_ms"]:
        reasons.append(("gap", "The trace paused or dropped points. Draw in one smooth motion."))
    if sp["pen_time_s"] < g["min_duration_s"]:
        reasons.append(
            (
                "too_short",
                f"The drawing lasted {sp['pen_time_s']:.1f} s; trace slowly for at least "
                f"{g['min_duration_s']:.0f} s.",
            )
        )
    if m["lifts"] > g["max_lifts"]:
        reasons.append(("lifts", "The finger lifted off the screen. Keep it down for the whole spiral."))
    if m["speed_mm_s"] > g["max_speed_mm_s"]:
        reasons.append(("too_fast", "Drawing too fast to measure. Trace slowly, like writing neatly."))
    if "coverage" in m and m["coverage"] < g["min_coverage"]:
        reasons.append(("coverage", f"Only {m['coverage'] * 100:.0f}% of the spiral was traced. Finish it."))
    if "template_dev_mm" in m and m["template_dev_mm"] > g["max_template_dev_mm"]:
        reasons.append(("off_template", "The trace doesn't follow the printed spiral. Stay on the line."))
    return [{"code": c, "message": msg} for c, msg in reasons]


def analyze(t_ms, x_mm, y_mm, stroke=None, template=None, gate_cfg=None):
    """Product entry point: gate first, numbers only if the gate passes."""
    m = compute(t_ms, x_mm, y_mm, stroke, template)
    reasons = gate(m, gate_cfg)
    ok = not reasons
    dominant = m["prominence"] >= MIN_PROMINENCE and m["amp_pp_mm"] >= MIN_RHYTHM_PP_MM
    return {
        "status": "ok" if ok else "redo",
        "reasons": reasons,
        "quality": {
            "rate_hz": m["sampling"]["rate_hz"],
            "max_gap_ms": m["sampling"]["max_gap_ms"],
            "duration_s": m["sampling"]["pen_time_s"],
            "lifts": m["lifts"],
            "speed_mm_s": m["speed_mm_s"],
            "coverage": m.get("coverage"),
            "template_dev_mm": m.get("template_dev_mm"),
        },
        "metrics": None
        if not ok
        else {
            "freq_hz": m["freq_hz"] if dominant else None,
            "dominant_rhythm": bool(dominant),
            "rule": {"min_prominence": MIN_PROMINENCE, "min_rhythm_pp_mm": MIN_RHYTHM_PP_MM},
            "prominence": m["prominence"],
            "amp_pp_mm": m["amp_pp_mm"],
            "rms_mm": m["rms_mm"],
            "speed_mm_s": m["speed_mm_s"],
        },
        "series": m["series"],
        "center": m["center"],
    }


def series_for_plot(series, max_points=600):
    """Thin the arrays for drawing on a canvas."""
    out = {}
    for k, v in series.items():
        v = np.asarray(v)
        step = max(1, v.size // max_points) if k not in ("freqs", "psd") else 1
        out[k] = [round(float(z), 5) for z in v[::step]]
    return out
