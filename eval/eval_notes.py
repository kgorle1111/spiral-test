"""Does the number guard hold? Draft n notes from synthetic record sets and count breaches.

Synthetic inputs only: this tests the guard + fallback plumbing, NOT clinical validity
or note quality on real patient data (untested on real data).

Run: uv run --extra note python eval/eval_notes.py [n] [seed]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from spiral import note
from spiral.note import draft_note, number_guard
from spiral.trend import summarize, table_for_note

TODAY = "2026-11-16"
TOKENS_PER_NOTE = 1200  # ~1k in + ~0.2k out, plus headroom for low-effort thinking
TAGS = ["caffeine", "poor sleep", "new dose", "stress", "alcohol", "late dose"]
OUT = Path(__file__).resolve().parent.parent / "results" / "notes_eval.json"


def synth_records(rng: np.random.Generator) -> list[dict]:
    base = float(np.exp(rng.uniform(np.log(3), np.log(40))))
    ratio = float(np.exp(rng.uniform(np.log(0.35), np.log(1.6))))
    sd = float(rng.uniform(0.03, 0.45))
    hand = "right" if rng.random() < 0.7 else "left"
    use_freq = rng.random() < 0.85
    tags_pool = list(rng.choice(TAGS, size=int(rng.integers(0, 4)), replace=False))
    recs = []
    for week_start, level in ((13, base), (6, base * ratio)):
        n = int(rng.integers(2, 10))  # some weeks fall under the 3-drawing minimum
        for off in sorted(rng.choice(7, size=min(n, 7), replace=False)):
            d = date.fromisoformat(TODAY) - timedelta(days=week_start - int(off))
            recs.append(
                {
                    "date": d.isoformat(),
                    "time": "09:00",
                    "amp_pp_mm": float(level * np.exp(rng.normal(0, sd))),
                    "freq_hz": float(rng.uniform(4, 9)) if use_freq else None,
                    "speed_mm_s": float(rng.uniform(10, 40)),
                    "tags": [t for t in tags_pool if rng.random() < 0.5],
                    "hand": hand,
                }
            )
    return recs


def eval_notes(n: int = 50, seed: int = 0) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("skipped: ANTHROPIC_API_KEY not set")
        return None
    est = n * TOKENS_PER_NOTE
    print(
        f"estimated cost: {n} notes x ~{TOKENS_PER_NOTE} tokens = ~{est:,} tokens "
        f"(~${est / 1e6 * 0.5:.2f} at the output rate of {note.MODEL})"
    )
    import anthropic  # needs the `note` extra

    client = anthropic.Anthropic(timeout=note.TIMEOUT_S, max_retries=2)
    rng = np.random.default_rng(seed)
    cases, sources = [], Counter()
    first_draft_breach_cases = final_breaches = errors = 0
    for i in range(n):
        recs = synth_records(rng)
        hand = recs[0]["hand"]
        summary = summarize(recs, TODAY)
        r = draft_note(summary, client=client, hand=hand)
        final = number_guard(r["text"], table_for_note(summary, hand))
        final_breaches += len(final)
        first_draft_breach_cases += bool(r["breaches"])
        errors += "error" in r
        sources[r["source"]] += 1
        cases.append(
            {
                "case": i,
                "source": r["source"],
                "first_draft_breaches": r["breaches"],
                "retry_breaches": r.get("retry_breaches", []),
                "error": r.get("error"),
            }
        )
        print(f"{i + 1}/{n} {r['source']:8} first-draft breaches: {r['breaches']}")
    result = {
        "model": note.MODEL,
        "n": n,
        "seed": seed,
        "data": "synthetic (untested on real data)",
        "final_output_breaches": final_breaches,
        "cases_with_first_draft_breach": first_draft_breach_cases,
        "sources": dict(sources),
        "api_errors": errors,
        "cases": cases,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(
        f"final-output breaches: {final_breaches} (must be 0); "
        f"first drafts caught: {first_draft_breach_cases}/{n}; wrote {OUT}"
    )
    return result


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    res = eval_notes(*args)
    sys.exit(1 if res and res["final_output_breaches"] else 0)
