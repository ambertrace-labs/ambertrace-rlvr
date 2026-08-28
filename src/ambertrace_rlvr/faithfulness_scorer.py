"""Rich scoring for faithfulness-under-optimization experiments (#50).

``score_batch_rich`` captures the full :class:`RichScore` per completion (reward,
reasoning, credited rules, consistency) — everything the faithfulness harness
needs to track monitorability over training steps.

``append_trajectory`` writes :func:`~ambertrace_rlvr.faithfulness.load_trajectory`
-compatible JSONL lines (one per completion), so a training loop can stream its
trajectory to disk and the curve analysis just reads it back.

Fail-closed: a parse failure resolves to a floor ``RichScore`` (floor reward,
empty reasoning/credited_rules, zero consistency). Never raises into the caller.

Extension points:

* Supply a custom :class:`~ambertrace_rlvr.rewards.RewardShaper` for different
  reward compositions.
* The ``verifier`` parameter accepts any object satisfying the
  :class:`~ambertrace_rlvr.evaluation.VerifierLike` protocol (including
  :class:`~ambertrace_rlvr.testing.FakeVerifier` for offline dev).
* Plug in a custom :class:`~ambertrace_rlvr.parsers.CompletionParser` for
  domain-specific decision-block formats.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_oracle import OracleJudgment
from .evaluation import VerifierLike
from .parsers import CompletionParser, ParsedCompletion
from .reports import AmberReport
from .rewards import RewardShaper, reasoning_consistency
from .verifier import score_one_item

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RichScore:
    """Per-completion score with the faithfulness-relevant fields.

    * ``reward`` — the shaped scalar reward (from the shaper's total).
    * ``reasoning`` — the model's stated reasoning (from the parser).
    * ``credited_rules`` — the rules the certificate credited for the decision
      (from :class:`~ambertrace_rlvr.eval_oracle.OracleJudgment`).
    * ``consistency`` — reasoning-vs-certified-trace precision metric
      (from :func:`~ambertrace_rlvr.rewards.reasoning_consistency`):
      fired-rule hits minus unfired-rule false claims, over the fired set.

    Note: ``consistency`` is a **precision-side** metric (does the reasoning
    accurately reflect the certified derivation?).  The trajectory's
    ``faithfulness`` (computed by :func:`~ambertrace_rlvr.faithfulness.faithfulness`)
    is the **recall-side** metric (what fraction of credited rules does the
    reasoning cite?).  Both are persisted when
    :func:`append_trajectory` writes a JSONL line.
    """

    reward: float
    reasoning: str
    credited_rules: tuple[str, ...]
    consistency: float


def consistency_score(parsed: ParsedCompletion, report: AmberReport) -> float:
    """Thin alias for :func:`~ambertrace_rlvr.rewards.reasoning_consistency`.

    Kept for backward compatibility with code that imported from this module
    before the canonical implementation was extracted into ``rewards.py``.
    """
    return reasoning_consistency(parsed, report)


def score_batch_rich(
    parser: CompletionParser,
    shaper: RewardShaper,
    verifier: VerifierLike,
    prompts: Sequence[str],
    completions: Sequence[str],
    metadata: Sequence[dict[str, Any]] | None = None,
    *,
    floor: float = -1.0,
) -> list[RichScore]:
    """Score a batch of completions, returning rich per-completion data.

    The parse -> verify -> shape loop delegates to
    :func:`~ambertrace_rlvr.verifier.score_one_item`, the shared per-item
    helper also used by :func:`build_reward_function`, so the two paths cannot
    diverge.

    Parameters
    ----------
    parser:
        Extracts the decision payload from each completion.
    shaper:
        Turns (parsed, report) into a scalar reward.
    verifier:
        Any object satisfying ``VerifierLike`` (``verify_batch``).
    prompts / completions:
        Parallel sequences of prompt + completion strings.
    metadata:
        Optional per-completion metadata dicts (``gold``, ``criteria_gold``).
    floor:
        Reward floor for unparseable / unverifiable completions.

    Returns
    -------
    list[RichScore]
        One ``RichScore`` per completion, in order.
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
            scores.append(_floor_score(floor))
            continue
        try:
            reward = score_one_item(shaper, pc, report, m, floor)
            judgment = OracleJudgment.from_report(report)
            scores.append(RichScore(
                reward=reward,
                reasoning=pc.reasoning or "",
                credited_rules=judgment.credited_rules,
                consistency=reasoning_consistency(pc, report),
            ))
        except Exception:
            logger.exception("rich scoring failed; flooring")
            scores.append(_floor_score(floor))
    return scores


def append_trajectory(
    path: str | Path,
    step: int,
    scores: Sequence[RichScore],
) -> None:
    """Append JSONL lines compatible with :func:`faithfulness.load_trajectory`.

    Each line is ``{step, reasoning, reward, credited_rules, consistency}`` --
    one per completion in ``scores``.  ``consistency`` is persisted alongside
    the recall-side ``faithfulness`` signal so both metrics are available during
    analysis (``load_trajectory`` ignores unknown fields).

    Round-trip: ``append_trajectory`` -> ``load_trajectory`` -> ``faithfulness_curve``.
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
                "consistency": s.consistency,
            })
            f.write(line + "\n")


def _floor_score(floor: float) -> RichScore:
    """A fail-closed floor score: the explicit floor reward, empty
    reasoning/credited_rules, zero consistency."""
    return RichScore(
        reward=floor,
        reasoning="",
        credited_rules=(),
        consistency=0.0,
    )
