from types import SimpleNamespace

import pytest

from spiral import note
from spiral.note import draft_note, number_guard
from spiral.trend import DISCLAIMER, summarize, table_for_note

TABLE = {
    "Amplitude": "12.3 mm",
    "Frequency": "5.2 Hz",
    "Change": "21% lower",
    "Date": "2026-11-16",
    "Drawings": "4",
    "Speed": "20.5 mm/s",
}


@pytest.mark.parametrize(
    "text",
    [
        "6 Hz",
        "6.1Hz",
        "0.9mm",
        "0.9 mm",
        "38%",
        "38 percent",
        "thirty-eight percent",
        "thirty eight percent",
        "six hertz",
        "half",
        "doubled",
        "twice the size",
        "1/2",
        "½",
        "−0.5",
        "-0.5",
        "3x",
        "3×",
        "three times",
        "4–6 Hz",
        "4-6 Hz",
        "2026-01-01",
        "forty-two",
        "seven",
        "one hundred",
        "100",
        "zero",
        "nineteen",
        "a hundred",
        "٦ Hz",
        "5.2 mm",
        "12.3 Hz",
        "4.0x",
    ],
)
def test_guard_catches(text):
    assert number_guard(f"The tremor was {text} today.", TABLE) != []


@pytest.mark.parametrize(
    "text",
    [
        "Tremor amplitude was lower this week and the rhythm was steady.",
        "Amplitude 12.3 mm, frequency 5.2 Hz.",
        "Amplitude 12.3mm at 5.2Hz.",
        "A 21% change; twenty-one percent; 21 percent.",
        "Drawn on 2026-11-16 from 4 drawings; four drawings.",
        "Speed 20.5 mm/s.",
        f"Fine. {DISCLAIMER}",
    ],
)
def test_guard_passes_supported(text):
    assert number_guard(text, TABLE) == []


DIRECTION = {"Amplitude change": "38% lower", "Minimum": "3", "Drawings": "7"}


@pytest.mark.parametrize(
    "text",
    [
        "Amplitude 0.52 cm",
        "Frequency 5.2 kHz",
        "5.2 ms period",
        "0.52 m",
        "12.3 mg",
        "5.2Hzish",
        "the 2nd week 7th",
    ],
)
def test_guard_rejects_unknown_units(text):
    table = {"a": "0.52 mm", "b": "5.2 Hz", "c": "12.3", "d": "2", "e": "7"}
    assert number_guard(text, table) != []


def test_guard_unitless_words_still_ok():
    assert number_guard("4 drawings, 4 per week.", {"a": "4"}) == []


@pytest.mark.parametrize(
    "text", ["A 3-fold change.", "a three-fold change", "3 fold", "three-times larger", "3\u2011fold"]
)
def test_guard_dashed_multipliers(text):
    assert number_guard(text, DIRECTION) != []


@pytest.mark.parametrize(
    "text", ["Fell by a third.", "two thirds smaller", "a fifth lower", "one tenth", "three quarters"]
)
def test_guard_ordinal_fractions(text):
    assert number_guard(text, DIRECTION) != []


@pytest.mark.parametrize(
    "text",
    [
        "Tremor rose 38%.",
        "Amplitude increased by 38 percent.",
        "It is higher by 38%.",
        "-38%",
        "+38%",
        "Up 38%.",
    ],
)
def test_guard_direction_contradiction_lower(text):
    assert number_guard(text, DIRECTION) != []


@pytest.mark.parametrize(
    "text", ["Tremor fell 38%.", "Amplitude decreased by 38%.", "Down 38 percent.", "-38%"]
)
def test_guard_direction_contradiction_higher(text):
    assert number_guard(text, {"Amplitude change": "38% higher"}) != []


def test_guard_direction_consistent_passes():
    assert number_guard("Tremor size is 38% lower than before.", DIRECTION) == []
    assert number_guard("Amplitude fell 38%. It dropped by thirty-eight percent.", DIRECTION) == []
    assert number_guard("Amplitude is 38% higher.", {"Amplitude change": "38% higher"}) == []
    # contradiction is judged per sentence
    assert number_guard("Frequency rose. Amplitude was 38% lower.", DIRECTION) == []


def test_guard_range_with_supported_endpoints_is_fine_but_unit_must_match():
    t = {"a": "4 Hz", "b": "6 Hz"}
    assert number_guard("between 4-6 Hz", t) == []
    assert number_guard("between 4-6 mm", t) != []


def test_guard_reports_the_offending_token():
    assert number_guard("Fell 38% to 6.1Hz at 12.3 mm", TABLE) == ["38%", "6.1Hz"]


def test_guard_table_word_makes_word_supported():
    assert number_guard("about half", {"x": "half the baseline"}) == []


# ---------------------------------------------------------------- draft_note


class FakeClient:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        t = self.texts.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], stop_reason="end_turn")


@pytest.fixture
def summary():
    recs = []
    for i in range(7):
        for day, amp in ((13 - i, 20.0 + i * 0.3), (6 - i, 10.0 + i * 0.2)):
            recs.append(
                {
                    "date": f"2026-11-{16 - day:02d}",
                    "time": "09:00",
                    "amp_pp_mm": amp,
                    "freq_hz": 5.2,
                    "speed_mm_s": 20.0,
                    "tags": ["caffeine"],
                    "hand": "right",
                }
            )
    return summarize(recs, "2026-11-16")


def test_two_breaches_fall_back_to_template(summary):
    c = FakeClient("Amplitude fell by 99%.", "Now it fell 98%.")
    r = draft_note(summary, client=c)
    assert r["source"] == "template" and len(c.calls) == 2
    assert r["breaches"] == ["99%"] and r["retry_breaches"] == ["98%"]
    assert number_guard(r["text"], table_for_note(summary)) == []
    assert "99" not in r["text"] and r["text"].endswith(DISCLAIMER)


def test_one_breach_then_clean_returns_llm(summary):
    table = table_for_note(summary)
    clean = f"Amplitude {table['Geometric-mean amplitude, current week']} this week."
    c = FakeClient("It halved.", clean)
    r = draft_note(summary, client=c)
    assert r["source"] == "llm" and r["text"] == clean and r["breaches"] == ["halved"]
    assert "halved" in c.calls[1]["messages"][-1]["content"]  # breach list fed back
    assert r["model"] == note.MODEL


def test_clean_first_draft_makes_one_call(summary):
    c = FakeClient("Tremor size changed this week.")
    r = draft_note(summary, client=c)
    assert r["source"] == "llm" and r["breaches"] == [] and len(c.calls) == 1


def test_prompt_contains_table_numbers_only_from_table(summary):
    c = FakeClient("ok")
    draft_note(summary, client=c)
    sent = c.calls[0]["messages"][0]["content"]
    assert "Geometric-mean amplitude, current week" in sent and "ANTHROPIC" not in sent
    assert c.calls[0]["model"] == note.MODEL and "temperature" not in c.calls[0]


def test_api_error_falls_back_without_leaking_message(summary):
    class Boom:
        messages = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("sk-ant-SECRET")))

    r = draft_note(summary, client=Boom())
    assert r["source"] == "template" and r["error"] == "RuntimeError" and "SECRET" not in str(r)


def test_refusal_or_truncation_falls_back(summary):
    class Trunc(FakeClient):
        def create(self, **kw):
            r = super().create(**kw)
            r.stop_reason = "max_tokens"
            return r

    assert draft_note(summary, client=Trunc("cut off"))["source"] == "template"


def test_no_key_no_client_is_template_and_makes_no_call(summary, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = draft_note(summary)
    assert r["source"] == "template" and r["model"] is None
