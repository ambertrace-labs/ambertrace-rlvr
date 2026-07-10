"""The verifier: turns completions into rewards via an AmberTrace platform.

``AmberVerifier`` calls ``platforms.query`` through the public ``ambertraceai`` SDK,
normalises the response to an :class:`AmberReport`, and hands it to the shaper.
Design invariants:

* **Fail-closed** — a parse failure, SDK error, or timeout resolves to a floor
  reward. The returned reward function NEVER raises into the training loop.
* **Batched + bounded concurrency** — RL issues many verifications per step.
* **Content-addressed cache** — identical (platform, query, facts) hit the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .domain import VerifiableDomain
from .parsers import CompletionParser, ParsedCompletion
from .reports import AmberReport
from .rewards import DefaultRewardShaper, RewardShaper

logger = logging.getLogger(__name__)

# reward_fn(prompts, completions, metadata) -> list[float]
RewardFunction = Callable[..., list[float]]


def build_reward_function(
    parser: CompletionParser,
    shaper: RewardShaper,
    verify_batch: Callable[[list[ParsedCompletion | None]], list[AmberReport | None]],
    floor: float,
) -> RewardFunction:
    """Assemble a batch reward function shared by real and fake verifiers.

    Unparseable completions get ``floor`` without a verify call; parsed ones are
    verified and shaped. Never raises.
    """

    def reward_fn(prompts: Sequence[str], completions: Sequence[str],
                  metadata: Sequence[dict[str, Any]] | None = None,
                  **_: Any) -> list[float]:
        meta = list(metadata) if metadata is not None else [{}] * len(completions)
        parsed = [parser.parse(p, c) for p, c in zip(prompts, completions)]
        reports = verify_batch(parsed)
        rewards: list[float] = []
        for pc, report, m in zip(parsed, reports, meta):
            if pc is None or report is None:
                rewards.append(floor)
                continue
            try:
                gold = m.get("gold") if isinstance(m, dict) else None
                rewards.append(shaper.score(pc, report, gold).total)
            except Exception:  # shaping must never crash the loop
                logger.exception("reward shaping failed; flooring")
                rewards.append(floor)
        return rewards

    return reward_fn


@dataclass
class AmberVerifier:
    domain: VerifiableDomain
    shaper: RewardShaper = field(default_factory=DefaultRewardShaper)
    batch_size: int = 32
    max_concurrency: int = 16
    cache: bool = True
    floor: float = -1.0

    _client: Any = field(default=None, init=False, repr=False)
    _cache: dict[str, AmberReport] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _api(self) -> Any:
        if self._client is None:
            import ambertraceai  # lazy: keep import off the offline-test path
            self._client = ambertraceai.AmbertraceAPI(
                base_url=self.domain.base_url, api_key=self.domain.api_key,
            )
        return self._client

    def verify_one(self, parsed: ParsedCompletion) -> AmberReport:
        """Verify a single parsed completion. Fail-closed — always returns a report."""
        key = _cache_key(self.domain.platform_id, parsed)
        if self.cache:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None:
                return hit
        report = self._query(parsed)
        if self.cache:
            with self._lock:
                self._cache[key] = report
        return report

    def _query(self, parsed: ParsedCompletion) -> AmberReport:
        try:
            import ambertraceai
            result = self._api().platforms.query(
                self.domain.platform_id,
                query=parsed.query,
                facts=parsed.facts,
                relations=parsed.relations,
                explain=True,
            )
            return AmberReport.from_query_result(result)
        except ambertraceai.AmbertraceError as err:  # certification/gate failure
            logger.info("query fail-closed for platform %s: %s",
                        self.domain.platform_id, err)
            return AmberReport.from_error(err)
        except Exception as err:  # network/unexpected — still fail closed
            logger.exception("verifier error; flooring")
            return AmberReport.floor(reason=f"verifier_error: {err!r}")

    def verify_batch(
        self, parsed: list[ParsedCompletion | None]
    ) -> list[AmberReport | None]:
        """Verify a batch with bounded concurrency, preserving order. ``None`` in
        maps to ``None`` out (unparseable → no verify)."""
        results: list[AmberReport | None] = [None] * len(parsed)
        todo = [(i, pc) for i, pc in enumerate(parsed) if pc is not None]
        if not todo:
            return results
        workers = max(1, min(self.max_concurrency, len(todo)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, report in zip(
                (i for i, _ in todo),
                pool.map(lambda ip: self.verify_one(ip[1]), todo),
            ):
                results[i] = report
        return results

    def as_reward_function(self) -> RewardFunction:
        return build_reward_function(
            self.domain.parser, self.shaper, self.verify_batch, self.floor,
        )


def _cache_key(platform_id: int, parsed: ParsedCompletion) -> str:
    payload = json.dumps(
        {"pid": platform_id, "q": parsed.query,
         "facts": parsed.facts, "relations": parsed.relations},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
