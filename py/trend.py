"""Week-over-week tremor trend: "did my new dose help?"

Constraint: numpy + stdlib only, no imports from this package. This exact file is
loaded by the web app under Pyodide.

Method (per hand). Tremor amplitude is log-normal, so everything is done on
x = ln(amp_pp_mm) and reported back as geometric means.

    baseline = the 7 days before the comparison window
    current  = the last 7 days ending `today`
    day mean = mean of x over that day's drawings; the days are the samples
               (drawings on one day are not independent: same tremor state, same sitting)
    d        = mean(day means, cur) - mean(day means, base)   (log ratio, current / baseline)
    sem      = pooled SD of the day means (ddof=1, both weeks)
    n_base, n_cur = days with at least one drawing in each week
    mdc_own  = 1.96 * sqrt(sem^2/n_base + sem^2/n_cur)
    floor    = ln(1 + 0.51 / sqrt(min(n_base, n_cur)))
    needed   = max(mdc_own, floor)
    changed  = |d| > needed

The floor scales the published single-measurement minimal detectable change for
tablet spiral peak-to-peak amplitude (51% of baseline; Elble & Ellenbogen 2017,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5618112/) to a mean of n days by
1/sqrt(n). It stops a lucky low-variance week from calling a small change real.
Fewer than 3 drawing days in either week: no verdict.

Amplitude depends on the device's sample rate (linear interpolation attenuates an
11 Hz tremor about 16% at 45 points/s vs 120). When the two weeks' median
`quality.rate_hz` differ by more than 25%, the change block carries a `warning`.

kn: SEM is estimated from the person's own two weeks (n <= 14 points), so the
    own-variability limit is itself noisy; the floor is the backstop. Upgrade: SEM
    from a real phone test-retest study, then drop the self-estimate.

No diagnosis and no causal claims: tags are counted per week, never analysed.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np

MIN_PER_WEEK = 3
Z95 = 1.96
SINGLE_MDC = 0.51  # Elble & Ellenbogen 2017, fraction of baseline, one measurement
WEEK = 7
HANDS = ("right", "left")
MAX_TAGS_IN_NOTE = 3

DISCLAIMER = (
    "Research prototype. Not a medical device. Not for diagnosis. "
    "Method: spiral tracing on a phone browser; see receipt."
)

NOT_ENOUGH = f"not enough days with drawings yet (need {MIN_PER_WEEK} days per week)"
RATE_TOLERANCE = 0.25


# ---------------------------------------------------------------- input


def _clean_tag(tag) -> str:
    # Tags are free text typed by the user and later shown to an LLM: restrict
    # the alphabet and length so a tag cannot carry instructions.
    t = re.sub(r"[^a-z0-9 _-]", "", str(tag).lower())
    return re.sub(r"\s+", " ", t).strip()[:32]


def _parse(rec: dict):
    """-> (date, hand, amp, freq|None, tags, rate|None) or None when the record is unusable."""
    try:
        d = date.fromisoformat(str(rec["date"]))
        hand = rec["hand"]
        amp = float(rec["amp_pp_mm"])
    except (KeyError, TypeError, ValueError):
        return None
    if hand not in HANDS or not math.isfinite(amp) or amp <= 0:
        return None
    freq = rec.get("freq_hz")
    try:
        freq = None if freq is None else float(freq)
    except (TypeError, ValueError):
        freq = None
    if freq is not None and not (math.isfinite(freq) and freq > 0):
        freq = None
    tags = {t for t in (_clean_tag(x) for x in (rec.get("tags") or [])) if t}
    try:
        rate = float((rec.get("quality") or {}).get("rate_hz"))
    except (AttributeError, TypeError, ValueError):
        rate = None  # old records have no quality block
    if rate is not None and not (math.isfinite(rate) and rate > 0):
        rate = None
    return d, hand, amp, freq, tags, rate


# ---------------------------------------------------------------- stats


def _week(rows: list, start: date) -> dict:
    """Stats for the 7 days start..start+6. rows: parsed tuples already filtered."""
    days = [start + timedelta(days=i) for i in range(WEEK)]
    lo, hi = days[0], days[-1]
    inside = [r for r in rows if lo <= r[0] <= hi]
    rates = [r[5] for r in inside if r[5] is not None]
    freqs = [r[3] for r in inside if r[3] is not None]
    tags: dict[str, int] = {}
    for r in inside:
        for t in r[4]:
            tags[t] = tags.get(t, 0) + 1
    series, day_x = [], []
    for d in days:
        xs = [math.log(r[2]) for r in inside if r[0] == d]
        if xs:
            day_x.append(sum(xs) / len(xs))
        series.append(
            {
                "date": d.isoformat(),
                "n": len(xs),
                "mean_mm": round(math.exp(sum(xs) / len(xs)), 3) if xs else None,
            }
        )
    x = np.array(day_x, dtype=float)
    return {
        "start": lo.isoformat(),
        "end": hi.isoformat(),
        "n": len(inside),
        "n_days": sum(1 for s in series if s["n"]),
        "mean_mm": round(float(np.exp(x.mean())), 3) if x.size else None,  # geometric mean of day means
        "median_mm": round(float(np.median([r[2] for r in inside])), 3) if inside else None,
        "sd_log": round(float(x.std(ddof=1)), 4) if x.size > 1 else None,
        "freq_median_hz": round(float(np.median(freqs)), 2) if freqs else None,
        "rate_median_hz": round(float(np.median(rates)), 1) if rates else None,
        "series": series,
        "tags": dict(sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))),
        "_x": x,
    }


def _rate_warning(a, b) -> str | None:
    if a is None or b is None or max(a, b) / min(a, b) - 1.0 <= RATE_TOLERANCE:
        return None
    return (
        f"The two weeks were recorded at different sample rates ({a:.0f} vs {b:.0f} points/s); "
        "small changes may reflect the device, not the tremor."
    )


def _change(base: dict, cur: dict) -> dict:
    nb, nc = base["n_days"], cur["n_days"]
    warning = _rate_warning(base["rate_median_hz"], cur["rate_median_hz"])
    if nb < MIN_PER_WEEK or nc < MIN_PER_WEEK:
        return {
            "pct": None,
            "mdc_pct": None,
            "changed": False,
            "direction": "none",
            "verdict": NOT_ENOUGH,
            "basis": f"Drawing days: {nb} in the earlier week, {nc} in the latest week.",
            "warning": warning,
        }
    xb, xc = base["_x"], cur["_x"]
    pooled = ((nb - 1) * xb.var(ddof=1) + (nc - 1) * xc.var(ddof=1)) / (nb + nc - 2)
    sem = math.sqrt(pooled)
    mdc_own = Z95 * math.sqrt(sem**2 / nb + sem**2 / nc)
    floor = math.log1p(SINGLE_MDC / math.sqrt(min(nb, nc)))
    needed = max(mdc_own, floor)
    d = float(xc.mean() - xb.mean())
    pct = (math.exp(d) - 1.0) * 100.0
    changed = abs(d) > needed
    lower = d < 0

    # Needed change expressed on the side the data moved, as % of the earlier week.
    def as_pct(v):
        return (1.0 - math.exp(-v)) * 100.0 if lower else (math.exp(v) - 1.0) * 100.0

    mdc_pct = as_pct(needed)
    direction = ("lower" if lower else "higher") if changed else "none"
    chg, need = f"{abs(pct):.0f}%", f"{mdc_pct:.0f}%"
    if changed:
        verdict = (
            f"Tremor size is {chg} {direction} than the week before, "
            f"beyond normal day-to-day variation (needed: {need})."
        )
    else:
        verdict = f"Within normal day-to-day variation (change {chg}, needed {need})."
    basis = (
        f"Geometric means of daily averages over {nb} earlier and {nc} latest days; "
        "needed change is the larger of "
        f"your own day-to-day variation ({as_pct(mdc_own):.0f}%) and the published "
        f"floor ({as_pct(floor):.0f}%)."
    )
    return {
        "pct": round(pct, 1),
        "mdc_pct": round(mdc_pct, 1),
        "changed": changed,
        "direction": direction,
        "verdict": verdict,
        "basis": basis,
        "warning": warning,
    }


def summarize(records: list[dict], today: str | None = None) -> dict:
    """Per-hand baseline vs current comparison.

    `today` defaults to the latest record date (deterministic), or the system
    date when there are no usable records. Records with a bad date, hand or
    amplitude are skipped and counted in `skipped`.
    """
    parsed = [p for p in (_parse(r) for r in records) if p is not None]
    if today is not None:
        end = date.fromisoformat(today)
    elif parsed:
        end = max(p[0] for p in parsed)
    else:
        end = date.today()
    cur_start = end - timedelta(days=WEEK - 1)
    base_start = cur_start - timedelta(days=WEEK)
    hands = {}
    for h in HANDS:
        rows = [p for p in parsed if p[1] == h]
        base, cur = _week(rows, base_start), _week(rows, cur_start)
        change = _change(base, cur)
        for w in (base, cur):
            w.pop("_x")
        hands[h] = {"hand": h, "baseline": base, "current": cur, "change": change}
    return {"today": end.isoformat(), "hands": hands, "skipped": len(records) - len(parsed)}


# ---------------------------------------------------------------- text


def _hand_block(summary: dict, hand: str) -> dict:
    return summary["hands"][hand] if "hands" in summary else summary


def _mm(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f} mm" if v >= 10 else f"{v:.2f} mm"


def _hz(v) -> str:
    return "n/a" if v is None else f"{v:.1f} Hz"


def table_for_note(summary: dict, hand: str = "right") -> dict[str, str]:
    """Flat label -> value table. The only source of numbers for the LLM note.

    Accepts the full summarize() output (hand picks the block) or one hand block.
    """
    h = _hand_block(summary, hand)
    base, cur, ch = h["baseline"], h["current"], h["change"]
    t = {
        "Hand": h["hand"],
        "Date range": f"{base['start']} to {cur['end']}",
        "Previous week": f"{base['start']} to {base['end']}",
        "Current week": f"{cur['start']} to {cur['end']}",
        "Drawings, previous week": str(base["n"]),
        "Drawings, current week": str(cur["n"]),
        "Drawing days, previous week": str(base["n_days"]),
        "Drawing days, current week": str(cur["n_days"]),
        "Minimum drawing days per week": str(MIN_PER_WEEK),
        "Geometric-mean amplitude, previous week": _mm(base["mean_mm"]),
        "Geometric-mean amplitude, current week": _mm(cur["mean_mm"]),
        "Median frequency, previous week": _hz(base["freq_median_hz"]),
        "Median frequency, current week": _hz(cur["freq_median_hz"]),
    }
    if ch["pct"] is not None:
        side = "lower" if ch["pct"] < 0 else "higher"
        t["Amplitude change"] = f"{abs(ch['pct']):.0f}% {side}"
        t["Change needed to count"] = f"{ch['mdc_pct']:.0f}%"
    t["Change beyond normal variation"] = "yes" if ch["changed"] else "no"
    t["Verdict"] = ch["verdict"]
    if ch.get("warning"):
        t["Sample-rate warning"] = ch["warning"]
    for label, week in (("current", cur), ("previous", base)):
        for tag, n in list(week["tags"].items())[:MAX_TAGS_IN_NOTE]:
            t[f"Tag count, {label} week: {tag}"] = str(n)
    return t


def clinician_summary(records: list[dict], today: str | None = None, hand: str = "right") -> str:
    """Deterministic plain-text note for a neurologist. Numbers come from summarize()."""
    return render_clinician(summarize(records, today), hand)


def render_clinician(summary: dict, hand: str = "right") -> str:
    """clinician_summary() from an existing summarize() output (the LLM fallback path)."""
    h = _hand_block(summary, hand)
    base, cur, ch = h["baseline"], h["current"], h["change"]
    lines = [
        f"Spiral tracing, {h['hand']} hand, {base['start']} to {cur['end']}.",
        f"Drawings: {base['n']} in {base['start']} to {base['end']}, "
        f"{cur['n']} in {cur['start']} to {cur['end']}.",
    ]
    if base["n"] and cur["n"]:
        lines.append(
            f"Geometric-mean peak-to-peak amplitude: {_mm(base['mean_mm'])} then {_mm(cur['mean_mm'])}."
        )
        lines.append(f"Median frequency: {_hz(base['freq_median_hz'])} then {_hz(cur['freq_median_hz'])}.")
    lines.append(ch["verdict"][0].upper() + ch["verdict"][1:])
    tag_bits = [f"{tag} {n}" for tag, n in list(cur["tags"].items())[:MAX_TAGS_IN_NOTE]]
    if tag_bits:
        lines.append("Tags this week: " + ", ".join(tag_bits) + ".")
    if ch.get("warning"):
        lines.append(ch["warning"])
    lines.append(DISCLAIMER)
    return " ".join(lines)
