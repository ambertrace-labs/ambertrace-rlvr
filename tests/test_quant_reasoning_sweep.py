"""Reasoning-enabled quant sweep (#87): truncation detection, output
classification, resume/skip logic, and summary math. Offline stubs only."""

from __future__ import annotations

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.quant_reasoning_sweep import (
    ReasoningRecord,
    classify_output,
    is_truncated_reasoning,
    summarise_level,
    think_char_count,
)

# A 2-verb access domain: deny is restrictive (safety-critical), approve permissive.
V = (LabelSpec("deny", 0, True), LabelSpec("approve", 1, False))


def _item(id: str, oracle: str) -> DecisionItem:
    return DecisionItem(id=id, domain="test", prompt=f"case {id}",
                        vocabulary=V, oracle=oracle,
                        difficulty={"structure": "baseline"})


# ---- is_truncated_reasoning (inline <think> tags) --------------------------

def test_truncated_open_think_with_length():
    """An open <think> tag + finish_reason=length is truncation."""
    raw = "<think>Let me reason about this case carefully..."
    assert is_truncated_reasoning(raw, "length") is True


def test_not_truncated_when_think_closed():
    """A closed <think> block even with finish_reason=length is NOT truncation
    --- the model finished reasoning and got cut in the answer."""
    raw = "<think>reasoning here</think>app"
    assert is_truncated_reasoning(raw, "length") is False


def test_not_truncated_on_stop():
    """finish_reason=stop means the model chose to stop --- never truncation."""
    raw = "<think>open tag without close"
    assert is_truncated_reasoning(raw, "stop") is False


def test_not_truncated_no_think_tag():
    """No <think> tag at all, even with length, is not reasoning truncation."""
    raw = "I would choose to app"
    assert is_truncated_reasoning(raw, "length") is False


def test_truncated_case_insensitive():
    raw = "<Think>partial reasoning..."
    assert is_truncated_reasoning(raw, "length") is True


# ---- is_truncated_reasoning (separate reasoning_content field) -------------

def test_truncated_separate_reasoning_content():
    """Separate reasoning_content field + empty content + length = truncation."""
    assert is_truncated_reasoning(
        "", "length", reasoning_content="long reasoning trace here",
    ) is True


def test_not_truncated_separate_with_content():
    """Reasoning_content present but content has an answer --- not truncation
    (the model finished reasoning and produced an answer before budget ran out)."""
    assert is_truncated_reasoning(
        "deny", "length", reasoning_content="reasoning here",
    ) is False


def test_not_truncated_separate_on_stop():
    """finish_reason=stop with reasoning_content is a complete response."""
    assert is_truncated_reasoning(
        "", "stop", reasoning_content="reasoning here",
    ) is False


# ---- classify_output -------------------------------------------------------

def test_classify_truncated_inline():
    raw = "<think>working through the ratio... 80% of income is"
    bucket, answer = classify_output(raw, "length", ["deny", "approve"])
    assert bucket == "truncated"
    assert answer.answered is False
    assert answer.parse_ok is False


def test_classify_truncated_separate_field():
    """Truncation via separate reasoning_content field."""
    bucket, answer = classify_output(
        "", "length", ["deny", "approve"],
        reasoning_content="long reasoning trace about the policy...",
    )
    assert bucket == "truncated"
    assert answer.answered is False


def test_classify_decision_with_think():
    raw = "<think>The policy says deny when posture is out of date.</think>\ndeny"
    bucket, answer = classify_output(raw, "stop", ["deny", "approve"])
    assert bucket == "decision"
    assert answer.parse_ok is True
    assert answer.value == "deny"


def test_classify_decision_with_separate_reasoning():
    """Content has the answer, reasoning_content has the trace."""
    bucket, answer = classify_output(
        "deny", "stop", ["deny", "approve"],
        reasoning_content="The policy clearly states deny.",
    )
    assert bucket == "decision"
    assert answer.value == "deny"


def test_classify_refusal():
    raw = ""
    bucket, answer = classify_output(raw, "stop", ["deny", "approve"])
    assert bucket == "refusal"
    assert answer.answered is False


def test_classify_parse_fail():
    raw = "<think>hmm</think>\nI think maybe deny or approve, hard to say"
    bucket, answer = classify_output(raw, "stop", ["deny", "approve"])
    assert bucket == "parse_fail"
    assert answer.answered is True
    assert answer.parse_ok is False


def test_classify_decision_no_think():
    """A clean answer with no think block at all."""
    raw = "deny"
    bucket, answer = classify_output(raw, "stop", ["deny", "approve"])
    assert bucket == "decision"
    assert answer.value == "deny"


def test_classify_length_but_closed_think_with_answer():
    """Think closed, answer present, but hit length --- still a decision."""
    raw = "<think>short</think>\ndeny"
    bucket, answer = classify_output(raw, "length", ["deny", "approve"])
    assert bucket == "decision"
    assert answer.value == "deny"


# ---- think_char_count ------------------------------------------------------

def test_think_char_count_complete_inline():
    raw = "<think>hello world</think>\ndeny"
    assert think_char_count(raw) == len("hello world")


def test_think_char_count_truncated_inline():
    raw = "<think>partial reasoning without close"
    assert think_char_count(raw) == len("partial reasoning without close")


def test_think_char_count_multiple_inline():
    raw = "<think>first</think> middle <think>second</think> deny"
    assert think_char_count(raw) == len("first") + len("second")


def test_think_char_count_empty():
    assert think_char_count("deny") == 0


def test_think_char_count_separate_field():
    """When reasoning_content is provided, its length is returned directly."""
    assert think_char_count("deny", reasoning_content="reasoning trace") == len("reasoning trace")


def test_think_char_count_separate_field_overrides_inline():
    """Separate field takes precedence over inline tags."""
    raw = "<think>inline</think>"
    assert think_char_count(raw, reasoning_content="separate") == len("separate")


# ---- ReasoningRecord round-trip --------------------------------------------

def test_record_round_trip():
    rec = ReasoningRecord(
        item_id="r1", raw="deny", finish_reason="stop",
        bucket="decision", parsed_value="deny", oracle="deny", think_chars=42,
        reasoning_content="The policy says deny.",
    )
    d = rec.to_dict()
    rec2 = ReasoningRecord.from_dict(d)
    assert rec2.item_id == rec.item_id
    assert rec2.bucket == rec.bucket
    assert rec2.parsed_value == rec.parsed_value
    assert rec2.reasoning_content == rec.reasoning_content
    assert rec2.think_chars == 42


def test_record_round_trip_legacy_no_reasoning_content():
    """Records from before reasoning_content field still load."""
    d = {"item_id": "r1", "raw": "deny", "finish_reason": "stop",
         "bucket": "decision", "parsed_value": "deny", "oracle": "deny",
         "think_chars": 0}
    rec = ReasoningRecord.from_dict(d)
    assert rec.reasoning_content == ""


# ---- summarise_level -------------------------------------------------------

def _make_records_and_items():
    """Two restrictive-oracle items and two permissive-oracle items, with
    varied buckets: one truncated, one correct, one over-permit, one decision."""
    items = {
        "r1": _item("r1", "deny"),    # restrictive oracle
        "r2": _item("r2", "deny"),    # restrictive oracle
        "p1": _item("p1", "approve"),  # permissive oracle
        "p2": _item("p2", "approve"),  # permissive oracle
    }
    records = [
        # r1: truncated mid-think --- excluded from all scoring
        ReasoningRecord(item_id="r1", raw="", finish_reason="length",
                        bucket="truncated", parsed_value=None, oracle="deny",
                        think_chars=300, reasoning_content="long trace..."),
        # r2: decision "approve" on a deny-oracle item --- over_permit + fail-open
        ReasoningRecord(item_id="r2", raw="approve",
                        finish_reason="stop", bucket="decision",
                        parsed_value="approve", oracle="deny", think_chars=50,
                        reasoning_content="reasoning"),
        # p1: decision "approve" on an approve-oracle item --- correct
        ReasoningRecord(item_id="p1", raw="approve",
                        finish_reason="stop", bucket="decision",
                        parsed_value="approve", oracle="approve", think_chars=40,
                        reasoning_content="reasoning"),
        # p2: refusal
        ReasoningRecord(item_id="p2", raw="", finish_reason="stop",
                        bucket="refusal", parsed_value=None, oracle="approve",
                        think_chars=0),
    ]
    return records, items


def test_summarise_level_counts():
    records, items = _make_records_and_items()
    s = summarise_level("Q8_0", records, items)
    assert s.n_items == 4
    assert s.n_truncated == 1
    assert s.n_decision == 2
    assert s.n_refusal == 1
    assert s.n_parse_fail == 0


def test_summarise_level_truncated_excluded_from_scoring():
    """The truncated item (r1) must NOT appear in correct/over_permit/over_deny."""
    records, items = _make_records_and_items()
    s = summarise_level("Q8_0", records, items)
    # Only r2 (over_permit) and p1 (correct) are scored decisions
    assert s.correct == 1
    assert s.over_permit == 1
    assert s.over_deny == 0
    assert s.accuracy == 0.5  # 1 correct / 2 scored


def test_summarise_level_fail_open_restrictive():
    """Fail-open on the restrictive band: r2 is the only restrictive decision
    and it over-permits, so fail_open_restrictive = 1/1 = 100%."""
    records, items = _make_records_and_items()
    s = summarise_level("Q8_0", records, items)
    assert s.fail_open_restrictive_count == 1
    assert s.restrictive_n == 1
    assert s.fail_open_restrictive == 1.0


def test_summarise_level_signed_bias():
    records, items = _make_records_and_items()
    s = summarise_level("Q8_0", records, items)
    # 1 over_permit, 0 over_deny, 2 scored => (1-0)/2 = 0.5
    assert s.signed_bias == 0.5


def test_summarise_level_serialisable():
    records, items = _make_records_and_items()
    s = summarise_level("Q8_0", records, items)
    d = s.to_dict()
    assert d["quant"] == "Q8_0"
    assert d["n_truncated"] == 1
    assert d["accuracy"] == 0.5


# ---- Resume/skip by item id -----------------------------------------------

def test_resume_skip_logic():
    """Simulate resume: given a set of already-completed item ids, only
    process the remaining ones."""
    all_ids = ["r1", "r2", "p1", "p2"]
    done_ids = {"r1", "r2"}
    remaining = [iid for iid in all_ids if iid not in done_ids]
    assert remaining == ["p1", "p2"]
