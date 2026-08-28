"""Rich scoring for faithfulness-under-optimization experiments (#50).

``score_batch_rich`` replicates the fail-closed parse -> verify -> shape loop
of :func:`~ambertrace_rlvr.verifier.build_reward_function` but captures the
full :class:`RichScore` per completion (reward, reasoning, credited rules,
consistency) — everything the faithfulness harness needs to track monitorability
over training steps.

``append_trajectory`` writes :func:`~ambertrace_rlvr.faithfulness.load_trajectory`
-compatible JSONL lines (one per completion), so a training loop can stream its
trajectory to disk and the curve analysis just reads it back.

Fail-closed: a parse failure resolves to a floor ``RichScore`` (floor reward,
empty reasoning/credited_rules, zero consistency). Never raises into the caller.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_oracle import OracleJudgment
from .parsers import CompletionParser, ParsedCompletion
from .reports import AmberReport
from .rewards import RewardShaper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RichScore:
    """Per-completion score with the faithfulness-relevant fields."""

    reward: float
    reasoning: str
    credited_rules: tuple[str, ...]
    consistency: float


def consistency_score(parsed: ParsedCompletion, report: AmberReport) -> float:
    """Public, module-level consistency metric.

    Mirrors :meth:`DefaultRewardShaper._consistency` — the rule-checked
    agreement between the completion's stated reasoning and the certified
    derivation.  Extracted here so callers outside the shaper can compute it
    without reaching into a private method.

    Fail-closed: zero on an uncertified report, no rules, no fired rules, or
    no captured reasoning.
    """
    import re

    if not report.proof_checked or not report.rules:
        return 0.0
    reasoning = parsed.reasoning
    if not reasoning:
        return 0.0
    fired = report.rules_fired
    if not fired:
        return 0.0
    text = reasoning.lower()

    def _names_rule(name: str, txt: str) -> bool:
        needle = name.strip().lower()
        if not needle:
            return False
        return re.search(rf"\b{re.escape(needle)}\b", txt) is not None

    fired_named = sum(1 for r in fired if _names_rule(r.name, text))
    unfired_named = sum(
        1 for r in report.rules if not r.fired and _names_rule(r.name, text)
    )
    return max(0.0, min(1.0, (fired_named - unfired_named) / len(fired)))


def score_batch_rich(
    parser: CompletionParser,
    shaper: RewardShaper,
    verifier: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    metadata: Sequence[dict[str, Any]] | None = None,
) -> list[RichScore]:
    """Score a batch of completions, returning rich per-completion data.

    ``verifier`` must expose ``verify_batch(list[ParsedCompletion | None])
    -> list[AmberReport | None]`` (works with :class:`testing.FakeVerifier`).

    The parse -> verify -> shape loop matches
    :func:`~ambertrace_rlvr.verifier.build_reward_function` exactly; the only
    difference is that we capture the full ``RichScore`` rather than just the
    scalar reward.
    """
    meta: list[dict[str, Any]] = (
        list(metadata) if metadata is not None else [{}] * len(completions)
    )
    parsed: list[ParsedCompletion | None] = [
        parser.parse(p, c) for p, c in zip(prompts, completions)
    ]
    reports: list[AmberReport | None] = verifier.verify_batch(parsed)

    scores: list[RichScore] = []
    for pc, report, m in zip(parsed, reports, meta):
        if pc is None or report is None:
            # Parse failure or verify returned None -> floor
            scores.append(_floor_score(shaper))
            continue
        try:
            gold = m.get("gold") if isinstance(m, dict) else None
            criteria_gold = m.get("criteria_gold") if isinstance(m, dict) else None
            breakdown = shaper.score(pc, report, gold, criteria_gold=criteria_gold)
            judgment = OracleJudgment.from_report(report)
            scores.append(RichScore(
                reward=breakdown.total,
                reasoning=pc.reasoning or "",
                credited_rules=judgment.credited_rules,
                consistency=consistency_score(pc, report),
            ))
        except Exception:
            logger.exception("rich scoring failed; flooring")
            scores.append(_floor_score(shaper))
    return scores


def append_trajectory(
    path: str | Path,
    step: int,
    scores: Sequence[RichScore],
) -> None:
    """Append JSONL lines compatible with :func:`faithfulness.load_trajectory`.

    Each line is ``{step, reasoning, reward, credited_rules}`` — one per
    completion in ``scores``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for s in scores:
            line = json.dumps({
                "step": step,
                "reasoning": s.reasoning,
                "reward": s.reward,
                "credited_rules": list(s.credited_rules),
            })
            f.write(line + "\n")


def _floor_score(shaper: RewardShaper) -> RichScore:
    """A fail-closed floor score: the shaper's clip lower-bound (or -1.0),
    empty reasoning/credited_rules, zero consistency."""
    clip = getattr(shaper, "clip", (-1.0, 2.0))
    floor = clip[0] if isinstance(clip, tuple) else -1.0
    return RichScore(
        reward=floor,
        reasoning="",
        credited_rules=(),
        consistency=0.0,
    )
