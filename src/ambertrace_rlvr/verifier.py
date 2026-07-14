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
import random
import threading
import time
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

    # Retry/backoff for transient (network/timeout/5xx) SDK errors. A legitimate
    # ``AmbertraceError`` certification/gate deny is never retried.
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_max: float = 8.0

    # Circuit breaker over consecutive transient failures.
    breaker_threshold: int = 5
    breaker_cooldown: float = 30.0

    _client: Any = field(default=None, init=False, repr=False)
    _cache: dict[str, AmberReport] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # Injectable clocks — tests replace these so the suite adds no wall-clock
    # delay and can advance the breaker's cooldown deterministically.
    _sleep: Callable[[float], None] = field(default=time.sleep, init=False, repr=False)
    _monotonic: Callable[[], float] = field(default=time.monotonic, init=False, repr=False)

    # Circuit-breaker state, guarded by its own lock (``verify_batch`` fans out
    # across a thread pool).
    _breaker_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)
    _half_open_pending: bool = field(default=False, init=False, repr=False)

    # Logged at most once per verifier instance — see verify_batch.
    _logged_no_batch: bool = field(default=False, init=False, repr=False)

    def _api(self) -> Any:
        if self._client is None:
            import ambertraceai  # lazy: keep import off the offline-test path
            self._client = ambertraceai.AmbertraceAPI(
                base_url=self.domain.base_url, api_key=self.domain.api_key,
            )
        return self._client

    def _redact(self, text: str) -> str:
        """Strip the platform API key out of any string headed for a log or a
        floor ``reason`` — keys must never reach logs or run reports."""
        key = self.domain.api_key
        if key:
            return text.replace(key, "***REDACTED***")
        return text

    def _breaker_allow(self) -> bool:
        """Whether a call may reach the SDK right now. Also claims the single
        half-open trial slot when the cooldown has elapsed."""
        with self._breaker_lock:
            if self._consecutive_failures < self.breaker_threshold:
                return True
            if self._opened_at is None:
                return True
            elapsed = self._monotonic() - self._opened_at
            if elapsed < self.breaker_cooldown:
                return False
            if self._half_open_pending:
                return False
            self._half_open_pending = True
            return True

    def _record_success(self) -> None:
        """A normal outcome — including a valid ``AmbertraceError`` deny —
        resets the breaker to closed."""
        with self._breaker_lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_pending = False

    def _record_transient_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            self._half_open_pending = False
            if self._consecutive_failures >= self.breaker_threshold:
                was_open = self._opened_at is not None
                self._opened_at = self._monotonic()
                if not was_open:  # loud, once, at the open transition
                    logger.warning(
                        "verifier circuit breaker OPEN for %.0fs after %d "
                        "consecutive failures; flooring reward source",
                        self.breaker_cooldown, self._consecutive_failures,
                    )

    def verify_one(self, parsed: ParsedCompletion) -> AmberReport:
        """Verify a single parsed completion. Fail-closed — always returns a report."""
        key = _cache_key(self.domain.platform_id, parsed)
        if self.cache:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None:
                return hit
        report, cacheable = self._query(parsed)
        if self.cache and cacheable:
            with self._lock:
                self._cache[key] = report
        return report

    def _query(self, parsed: ParsedCompletion) -> tuple[AmberReport, bool]:
        """Query the platform with retry/backoff + circuit breaker. Returns
        ``(report, cacheable)`` — a transient-failure or breaker-open floor is
        never cacheable, so it can't poison the cache for this key."""
        if not self._breaker_allow():
            logger.info(
                "circuit breaker open for platform %s; flooring without SDK call",
                self.domain.platform_id,
            )
            return AmberReport.floor(reason="circuit_open"), False

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                import ambertraceai
                result = self._api().platforms.query(
                    self.domain.platform_id,
                    query=parsed.query,
                    facts=parsed.facts,
                    relations=parsed.relations,
                    explain=True,
                )
                self._record_success()
                return AmberReport.from_query_result(result), True
            except ambertraceai.AmbertraceError as err:  # certification/gate failure
                # A legitimate deny, not a transient failure: no retry, no
                # effect on the breaker beyond resetting it (this platform is
                # reachable and answering).
                logger.info(
                    "query fail-closed for platform %s: %s",
                    self.domain.platform_id, self._redact(str(err)),
                )
                self._record_success()
                return AmberReport.from_error(err), True
            except Exception as err:  # network/timeout/5xx — retryable, counts toward breaker
                last_err = err
                if attempt < self.max_retries:
                    jitter = random.uniform(0, self.backoff_base)
                    # Clamp last so jitter can't push the delay past backoff_max.
                    delay = min(self.backoff_max, self.backoff_base * (2 ** attempt) + jitter)
                    logger.info(
                        "transient verifier error (attempt %d/%d) for platform %s; "
                        "retrying in %.2fs: %s",
                        attempt + 1, self.max_retries + 1, self.domain.platform_id,
                        delay, self._redact(repr(err)),
                    )
                    self._sleep(delay)
                    continue
                # Do NOT log with exc_info here: the logging formatter would
                # render the raw exception (message + traceback), bypassing
                # _redact and potentially leaking the API key in an auth/URL error.
                logger.error(
                    "verifier error; retries exhausted; flooring: %s",
                    self._redact(repr(err)),
                )
                self._record_transient_failure()
                reason = self._redact(f"verifier_error: {err!r}")
                return AmberReport.floor(reason=reason), False

        # Unreachable (the loop above always returns), kept for exhaustiveness.
        reason = self._redact(f"verifier_error: {last_err!r}")
        return AmberReport.floor(reason=reason), False

    def _supports_batch(self) -> bool:
        """Capability gate, not a version check: does the wired SDK client expose
        a ``platforms.query_batch``? As of ``ambertraceai==1.0.5`` it does not
        (see issue #27). Building a batch payload path against an unpublished
        signature would mean guessing at the SDK surface, so that work is
        deferred until the platform ships it."""
        return hasattr(self._api().platforms, "query_batch")

    def verify_batch(
        self, parsed: list[ParsedCompletion | None]
    ) -> list[AmberReport | None]:
        """Verify a batch with bounded concurrency, preserving order. ``None`` in
        maps to ``None`` out (unparseable → no verify).

        No batch payload path is built here: ``ambertraceai==1.0.5`` exposes
        neither ``platforms.query_batch`` nor a compact/projection param on
        ``platforms.query``, and guessing an unpublished signature is out of
        scope (see issue #27). This gates on capability and
        falls back to the existing per-item ``ThreadPoolExecutor`` pool, which
        is already the correct bounded-concurrency mechanism for the published
        SDK surface."""
        results: list[AmberReport | None] = [None] * len(parsed)
        todo = [(i, pc) for i, pc in enumerate(parsed) if pc is not None]
        if not todo:
            return results
        if not self._supports_batch() and not self._logged_no_batch:
            self._logged_no_batch = True
            logger.debug(
                "platform has no query_batch; verifying per-item at "
                "max_concurrency=%d pending platform support (see issue #27)",
                self.max_concurrency,
            )
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
