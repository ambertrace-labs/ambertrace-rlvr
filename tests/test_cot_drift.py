"""Offline tests for cot_drift — pure functions, no network."""

from __future__ import annotations

from ambertrace_rlvr.cot_drift import (
    BACKTRACKING,
    HEDGING,
    VERIFIER_AWARENESS,
    ProbeTrace,
    channel_lengths,
    distinct_n,
    group_similarity,
    lexicon_rate,
    ngram_logodds_diff,
    think_stated_divergence,
    unsupported_fact_fraction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trace(
    item_id: str = "t1",
    think: str = "I think we should escalate",
    stated: str = "By the emergency rule we escalate",
    decision: str | None = "escalate",
    facts: dict[str, object] | None = None,
) -> ProbeTrace:
    return ProbeTrace(
        item_id=item_id, think=think, stated=stated,
        decision=decision, facts=facts,
    )


_EMPTY = _trace(think="", stated="", decision=None, facts=None)


# ---------------------------------------------------------------------------
# channel_lengths
# ---------------------------------------------------------------------------

class TestChannelLengths:
    def test_basic(self):
        corpus = [
            _trace(think="one two three", stated="a b"),
            _trace(think="four five", stated="c d e f"),
        ]
        stats = channel_lengths(corpus)
        assert stats.think_mean == 2.5  # (3 + 2) / 2
        assert stats.stated_mean == 3.0  # (2 + 4) / 2
        assert stats.think_median == 2.5
        assert stats.stated_median == 3.0

    def test_empty_corpus(self):
        stats = channel_lengths([])
        assert stats.think_mean == 0.0
        assert stats.stated_mean == 0.0

    def test_empty_channels(self):
        stats = channel_lengths([_EMPTY])
        # "".split() -> [] -> len 0
        assert stats.think_mean == 0.0
        assert stats.stated_mean == 0.0


# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------

class TestDistinctN:
    def test_unique_corpus(self):
        corpus = [
            _trace(think="alpha beta gamma delta epsilon"),
            _trace(think="zeta eta theta iota kappa"),
        ]
        # All trigrams are unique.
        assert distinct_n(corpus, n=3, channel="think") == 1.0

    def test_repetitive_corpus(self):
        # Same text repeated -> distinct ratio < 1.
        text = "the cat sat on the mat"
        corpus = [_trace(think=text), _trace(think=text)]
        ratio = distinct_n(corpus, n=3, channel="think")
        assert 0.0 < ratio < 1.0

    def test_empty_corpus(self):
        assert distinct_n([], n=3) == 0.0

    def test_short_text(self):
        # Text shorter than n -> no n-grams -> 0.0.
        corpus = [_trace(think="hi")]
        assert distinct_n(corpus, n=3) == 0.0

    def test_stated_channel(self):
        corpus = [_trace(stated="one two three four")]
        assert distinct_n(corpus, n=2, channel="stated") > 0.0


# ---------------------------------------------------------------------------
# group_similarity
# ---------------------------------------------------------------------------

class TestGroupSimilarity:
    def test_identical_strings(self):
        groups = [["the cat sat on the mat", "the cat sat on the mat"]]
        assert group_similarity(groups) == 1.0

    def test_completely_different(self):
        groups = [["aaa bbb ccc ddd", "xxx yyy zzz www"]]
        assert group_similarity(groups) == 0.0

    def test_empty_groups(self):
        assert group_similarity([]) == 0.0
        assert group_similarity([["solo"]]) == 0.0  # no pairs

    def test_multiple_groups(self):
        groups = [
            ["the cat sat on the mat", "the cat sat on the mat"],
            ["aaa bbb ccc ddd", "xxx yyy zzz www"],
        ]
        sim = group_similarity(groups)
        assert sim == 0.5  # avg of 1.0 and 0.0


# ---------------------------------------------------------------------------
# lexicon_rate
# ---------------------------------------------------------------------------

class TestLexiconRate:
    def test_all_present(self):
        lexicon = ("foo", "bar")
        corpus = [_trace(think="we see foo and bar here")]
        assert lexicon_rate(corpus, lexicon, channel="think") == 1.0

    def test_none_present(self):
        lexicon = ("foo", "bar")
        corpus = [_trace(think="nothing relevant")]
        assert lexicon_rate(corpus, lexicon, channel="think") == 0.0

    def test_partial(self):
        lexicon = ("foo", "bar")
        corpus = [_trace(think="only foo is here")]
        assert lexicon_rate(corpus, lexicon, channel="think") == 0.5

    def test_case_insensitive(self):
        lexicon = ("Verifier",)
        corpus = [_trace(think="the VERIFIER said so")]
        assert lexicon_rate(corpus, lexicon) == 1.0

    def test_word_boundary(self):
        # "score" should not match inside "scored" — wait, \b matches between
        # "scor" and "ed" differently. Actually \bscore\b matches "score" as
        # a standalone word; "scored" contains "score" at a non-boundary?
        # re.search(r"\bscore\b", "scored") -> None, because "d" follows.
        lexicon = ("score",)
        corpus = [_trace(think="they scored well")]
        # "score" has \b before s and after e, "scored" -> \bscored\b; "score\b" won't match
        assert lexicon_rate(corpus, lexicon) == 0.0

    def test_empty_corpus(self):
        assert lexicon_rate([], ("foo",)) == 0.0

    def test_empty_lexicon(self):
        assert lexicon_rate([_trace()], ()) == 0.0

    def test_shipped_lexicons_are_tuples(self):
        # Verify they are tuples (immutable) and non-empty.
        for lex in (VERIFIER_AWARENESS, HEDGING, BACKTRACKING):
            assert isinstance(lex, tuple)
            assert len(lex) > 0

    def test_multi_word_lexicon_term(self):
        lexicon = ("get credit",)
        corpus = [_trace(think="we should get credit for this")]
        assert lexicon_rate(corpus, lexicon) == 1.0


# ---------------------------------------------------------------------------
# think_stated_divergence
# ---------------------------------------------------------------------------

class TestThinkStatedDivergence:
    def test_no_concealment_no_flip(self):
        trace = _trace(
            think="By Classify Is Emergency we escalate",
            stated="By Classify Is Emergency we escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, ["Classify Is Emergency"])
        assert report.rules_in_think_only == ()
        assert report.decision_flip is False
        assert report.channel_overlap == 1.0  # identical text

    def test_concealment(self):
        trace = _trace(
            think="Classify Is Emergency fires here",
            stated="This looks bad",
            decision="escalate",
        )
        report = think_stated_divergence(trace, ["Classify Is Emergency"])
        assert "Classify Is Emergency" in report.rules_in_think_only

    def test_decision_flip_true(self):
        # Think concludes "clear" but decision is "escalate".
        trace = _trace(
            think="After analysis the answer is clear",
            stated="We should escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is True

    def test_decision_flip_false_when_consistent(self):
        trace = _trace(
            think="After analysis we must escalate",
            stated="escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is False

    def test_decision_flip_false_no_decision(self):
        # No emitted decision -> no flip.
        trace = _trace(think="thinking about clear", decision=None)
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is False

    def test_decision_flip_false_no_decision_word_in_think(self):
        trace = _trace(
            think="the sky is blue and birds fly",
            stated="escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is False

    def test_no_credited_rules(self):
        trace = _trace()
        report = think_stated_divergence(trace, [])
        assert report.rules_in_think_only == ()

    def test_empty_channels(self):
        report = think_stated_divergence(_EMPTY, ["R1"])
        assert report.rules_in_think_only == ()
        assert report.decision_flip is False
        # Both empty -> Jaccard of two empty sets is 1.0.
        assert report.channel_overlap == 1.0

    def test_last_decision_wins(self):
        # Think mentions "clear" then "escalate" -> last is "escalate".
        trace = _trace(
            think="first I thought clear but then I think escalate",
            stated="escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is False

    def test_unclear_does_not_match_clear(self):
        # Regression: "unclear" must NOT match as "clear" via substring.
        # Think ends with "...I will escalate. The IFF is unclear." with
        # decision "escalate" -> decision_flip must be False.
        trace = _trace(
            think="After reviewing the data I will escalate. The IFF is unclear.",
            stated="escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is False

    def test_true_decision_flip_still_detected(self):
        # A genuine flip (think says "clear", decision says "escalate")
        # must still be detected after the word-boundary fix.
        trace = _trace(
            think="Everything looks fine; the track is clear.",
            stated="escalate",
            decision="escalate",
        )
        report = think_stated_divergence(trace, [])
        assert report.decision_flip is True


# ---------------------------------------------------------------------------
# ngram_logodds_diff
# ---------------------------------------------------------------------------

class TestNgramLogOddsDiff:
    def test_identical_corpora(self):
        text = "the cat sat on the mat and the dog sat on the rug"
        corpus = [_trace(think=text)] * 5
        diff = ngram_logodds_diff(corpus, corpus, n=1, min_count=2)
        # Identical distributions -> all log-odds near zero.
        for _, lo in diff.rising:
            assert abs(lo) < 1e-9
        for _, lo in diff.falling:
            assert abs(lo) < 1e-9

    def test_distinct_corpora(self):
        a_text = "alpha alpha alpha alpha alpha alpha"
        b_text = "beta beta beta beta beta beta"
        corpus_a = [_trace(think=a_text)] * 3
        corpus_b = [_trace(think=b_text)] * 3
        diff = ngram_logodds_diff(corpus_a, corpus_b, n=1, min_count=5)
        # "beta" should be rising; "alpha" should be falling.
        rising_words = {w for w, _ in diff.rising}
        falling_words = {w for w, _ in diff.falling}
        assert "beta" in rising_words
        assert "alpha" in falling_words

    def test_empty_corpus(self):
        diff = ngram_logodds_diff([], [], n=1)
        assert diff.rising == []
        assert diff.falling == []

    def test_min_count_filtering(self):
        # With min_count=100, nothing passes the filter.
        corpus = [_trace(think="word " * 10)]
        diff = ngram_logodds_diff(corpus, corpus, n=1, min_count=100)
        assert diff.rising == []
        assert diff.falling == []

    def test_stated_channel(self):
        corpus_a = [_trace(stated="cat " * 10)] * 2
        corpus_b = [_trace(stated="dog " * 10)] * 2
        diff = ngram_logodds_diff(corpus_a, corpus_b, n=1, min_count=5, channel="stated")
        rising_words = {w for w, _ in diff.rising}
        assert "dog" in rising_words

    def test_no_overlap_and_sign_correct(self):
        # Regression: rising must contain only positive log-odds, falling only
        # negative, and the two lists must never overlap -- even when fewer
        # than 2*top_k n-grams survive filtering.
        a_text = "alpha alpha alpha beta beta beta"
        b_text = "beta beta beta gamma gamma gamma"
        corpus_a = [_trace(think=a_text)] * 3
        corpus_b = [_trace(think=b_text)] * 3
        diff = ngram_logodds_diff(corpus_a, corpus_b, n=1, top_k=25, min_count=2)

        rising_words = {w for w, _ in diff.rising}
        falling_words = {w for w, _ in diff.falling}

        # No overlap between rising and falling.
        assert not (rising_words & falling_words), (
            f"rising/falling overlap: {rising_words & falling_words}"
        )

        # All rising entries must have positive log-odds.
        for _, lo in diff.rising:
            assert lo > 0, f"rising entry has non-positive log-odds: {lo}"

        # All falling entries must have negative log-odds.
        for _, lo in diff.falling:
            assert lo < 0, f"falling entry has non-negative log-odds: {lo}"

        # "alpha" fell (present in a, absent in b) -> must be in falling, not rising.
        assert "alpha" in falling_words
        assert "alpha" not in rising_words

        # "gamma" rose (absent in a, present in b) -> must be in rising, not falling.
        assert "gamma" in rising_words
        assert "gamma" not in falling_words


# ---------------------------------------------------------------------------
# unsupported_fact_fraction
# ---------------------------------------------------------------------------

class TestUnsupportedFactFraction:
    def test_all_grounded(self):
        trace = _trace(facts={"sensor": "radar", "speed": 300})
        prompt = "The track was detected by radar at 300 knots."
        assert unsupported_fact_fraction(trace, prompt) == 0.0

    def test_all_fabricated(self):
        trace = _trace(facts={"sensor": "lidar", "speed": 999})
        prompt = "The track was detected by radar at 300 knots."
        frac = unsupported_fact_fraction(trace, prompt)
        assert frac == 1.0

    def test_partial(self):
        trace = _trace(facts={"sensor": "radar", "speed": 999})
        prompt = "The track was detected by radar at 300 knots."
        frac = unsupported_fact_fraction(trace, prompt)
        assert frac == 0.5

    def test_no_facts(self):
        trace = _trace(facts=None)
        assert unsupported_fact_fraction(trace, "anything") == 0.0

    def test_empty_facts(self):
        trace = _trace(facts={})
        assert unsupported_fact_fraction(trace, "anything") == 0.0
