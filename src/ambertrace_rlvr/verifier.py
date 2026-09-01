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

# The minimal set of top-level response fields ``AmberReport.from_query_result``
# needs.  Requesting only these via the SDK's ``projection`` parameter avoids
# transferring the full explanation payload on every query and ``query_batch``
# call.  The ``explanation`` field carries the sub-structure (confidence, rules,
# rejected facts, deciding rules, schema_version) that the report normaliser
# reads, so it must always be present.
REWARD_PROJECTION: list[str] = [
    "proof_checked",
    "answer",
    "decision",
    "proof_summary",
    "explanation",
]

# Maximum items per ``query_batch`` call (SDK + platform limit).
_BATCH_CHUNK_SIZE = 50


def score_one_item(
    shaper: RewardShaper,
    parsed: ParsedCompletion,
    report: AmberReport,
    meta: dict[str, Any],
    floor: float,
) -> float:
    """Shape one (parsed, report) pair into a scalar reward.

    This is the shared per-item step used by :func:`build_reward_function`
    and :func:`~ambertrace_rlvr.faithfulness_scorer.score_batch_rich`.
    Fail-closed: shaping errors resolve to ``floor``, never an exception.

    Parameters
    ----------
    shaper:
        Turns (parsed, report) into a :class:`RewardBreakdown`.
    parsed:
        The parsed completion (must not be ``None``).
    report:
        The certificate (must not be ``None``).
    meta:
        Per-item metadata dict (may contain ``gold``, ``criteria_gold``).
    floor:
        Reward floor on shaping failure.
    """
    try:
        gold = meta.get("gold") if isinstance(meta, dict) else None
        criteria_gold = meta.get("criteria_gold") if isinstance(meta, dict) else None
        return shaper.score(parsed, report, gold, criteria_gold=criteria_gold).total
    except Exception:
        logger.exception("reward shaping failed; flooring")
        return floor


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
            rewards.append(score_one_item(shaper, pc, report, m, floor))
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

    # When True (default) and the SDK supports ``projection``, request only the
    # minimal set of response fields ``AmberReport.from_query_result`` needs.
    # Set to False to always fetch the full response (debugging / audit).
    use_projection: bool = True

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

    def _projection_args(self) -> dict[str, Any]:
        """Return ``{"projection": [...]}`` when projection is enabled and the
        SDK supports it, else ``{}``.

        ``projection`` was added to both ``query`` and ``query_batch`` in SDK
        2.1.3 — the same release that introduced ``query_batch``.  So we gate
        on ``query_batch`` presence (already checked by ``_supports_batch``)
        as the proxy for projection support; an SDK without ``query_batch``
        also lacks ``projection``.
        """
        if not self.use_projection:
            return {}
        if not hasattr(self._api().platforms, "query_batch"):
            return {}
        return {"projection": list(REWARD_PROJECTION)}

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
                # ``predictions`` is passed only when present so the ordinary
                # facts-only reward path keeps working against any SDK build.
                extra: dict[str, Any] = {}
                if parsed.predictions is not None:
                    extra["predictions"] = parsed.predictions
                extra.update(self._projection_args())
                result = self._api().platforms.query(
                    self.domain.platform_id,
                    query=parsed.query,
                    facts=parsed.facts,
                    relations=parsed.relations,
                    explain=True,
                    **extra,
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
            except Exception as err:  # noqa: BLE001 — network/timeout/5xx — retryable, counts toward breaker
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
        a ``platforms.query_batch``?"""
        return hasattr(self._api().platforms, "query_batch")

    def _build_batch_query(self, parsed: ParsedCompletion) -> dict[str, Any]:
        """Build a single item dict for a ``query_batch`` call."""
        item: dict[str, Any] = {
            "query": parsed.query,
            "facts": parsed.facts,
            "explain": True,
        }
        if parsed.relations:
            item["relations"] = parsed.relations
        if parsed.predictions is not None:
            item["predictions"] = parsed.predictions
        return item

    def _query_batch_chunk(
        self,
        items: list[tuple[int, ParsedCompletion]],
    ) -> list[tuple[int, AmberReport, bool]]:
        """Execute one chunk (<=50) via ``query_batch`` with retry/backoff +
        circuit breaker. Returns ``[(original_index, report, cacheable), ...]``.

        A batch-level transport failure retries the entire batch and counts
        toward the breaker. Per-item errors within a successful batch response
        are handled individually: a certification/gate deny (status ``"error"``
        whose error matches a known deny pattern) produces ``from_error``
        (cacheable); any other per-item error produces a floor (not cacheable).
        One bad row never fails the batch.
        """
        if not self._breaker_allow():
            logger.info(
                "circuit breaker open for platform %s; flooring batch chunk "
                "without SDK call",
                self.domain.platform_id,
            )
            return [
                (idx, AmberReport.floor(reason="circuit_open"), False)
                for idx, _ in items
            ]

        queries = [self._build_batch_query(pc) for _, pc in items]
        proj_args = self._projection_args()

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                import ambertraceai
                resp = self._api().platforms.query_batch(
                    self.domain.platform_id,
                    queries=queries,
                    **proj_args,
                )
                self._record_success()
                # Parse per-item results in request order.
                results_list = resp.get("results", []) if isinstance(resp, dict) else []
                out: list[tuple[int, AmberReport, bool]] = []
                for i, (idx, _pc) in enumerate(items):
                    if i < len(results_list):
                        item_result = results_list[i]
                    else:
                        # Missing result — floor.
                        out.append(
                            (idx, AmberReport.floor(reason="missing_batch_item"), False)
                        )
                        continue
                    status = item_result.get("status") if isinstance(item_result, dict) else None
                    if status == "ok":
                        data = item_result.get("data", {})
                        out.append((idx, AmberReport.from_query_result(data), True))
                    elif status == "error":
                        err_body = item_result.get("error", {})
                        err_msg = err_body.get("message", "") if isinstance(err_body, dict) else str(err_body)
                        err_code = err_body.get("code", "") if isinstance(err_body, dict) else ""
                        # Certification/gate deny — treat like AmbertraceError.
                        # These have structured codes; a transport error at the
                        # item level would not have a code.
                        if err_code:
                            status_code = err_body.get("status_code", 422) if isinstance(err_body, dict) else 422
                            synth_err = ambertraceai.AmbertraceError(
                                status_code, err_code, err_msg,
                                rejected_facts=err_body.get("rejected_facts") if isinstance(err_body, dict) else None,
                            )
                            out.append(
                                (idx, AmberReport.from_error(synth_err), True)
                            )
                        else:
                            reason = self._redact(
                                f"batch_item_error: {err_msg}"
                            )
                            out.append(
                                (idx, AmberReport.floor(reason=reason), False)
                            )
                    else:
                        out.append(
                            (idx, AmberReport.floor(reason="unknown_batch_status"), False)
                        )
                return out
            except Exception as err:  # noqa: BLE001 — batch-level transport failure
                last_err = err
                if attempt < self.max_retries:
                    jitter = random.uniform(0, self.backoff_base)
                    delay = min(self.backoff_max, self.backoff_base * (2 ** attempt) + jitter)
                    logger.info(
                        "transient batch error (attempt %d/%d) for platform %s; "
                        "retrying in %.2fs: %s",
                        attempt + 1, self.max_retries + 1, self.domain.platform_id,
                        delay, self._redact(repr(err)),
                    )
                    self._sleep(delay)
                    continue
                logger.error(
                    "batch error; retries exhausted; flooring chunk: %s",
                    self._redact(repr(err)),
                )
                self._record_transient_failure()
                reason = self._redact(f"verifier_error: {last_err!r}")
                return [
                    (idx, AmberReport.floor(reason=reason), False)
                    for idx, _ in items
                ]

        # Unreachable.
        reason = self._redact(f"verifier_error: {last_err!r}")
        return [
            (idx, AmberReport.floor(reason=reason), False)
            for idx, _ in items
        ]

    def verify_batch(
        self, parsed: list[ParsedCompletion | None]
    ) -> list[AmberReport | None]:
        """Verify a batch with bounded concurrency, preserving order. ``None`` in
        maps to ``None`` out (unparseable -> no verify).

        When ``query_batch`` is available on the SDK, cache-misses are grouped
        into chunks of up to 50 and dispatched via the batch endpoint.  Chunks
        are fanned out across a thread pool for multi-chunk concurrency.  When
        ``query_batch`` is absent, falls back to per-item ``verify_one`` through
        the same ``ThreadPoolExecutor`` pool — the library still works against
        older servers/SDKs.
        """
        results: list[AmberReport | None] = [None] * len(parsed)
        todo: list[tuple[int, ParsedCompletion]] = []
        for i, pc in enumerate(parsed):
            if pc is None:
                continue
            if self.cache:
                key = _cache_key(self.domain.platform_id, pc)
                with self._lock:
                    hit = self._cache.get(key)
                if hit is not None:
                    results[i] = hit
                    continue
            todo.append((i, pc))
        if not todo:
            return results

        if self._supports_batch():
            # Chunk into <=50 and fan out chunks across a thread pool.
            chunks: list[list[tuple[int, ParsedCompletion]]] = []
            for start in range(0, len(todo), _BATCH_CHUNK_SIZE):
                chunks.append(todo[start:start + _BATCH_CHUNK_SIZE])

            workers = max(1, min(self.max_concurrency, len(chunks)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                chunk_results = list(pool.map(self._query_batch_chunk, chunks))

            for chunk_out in chunk_results:
                for idx, report, cacheable in chunk_out:
                    results[idx] = report
                    if self.cache and cacheable:
                        # Retrieve the parsed completion for caching.
                        pc_for_cache = parsed[idx]
                        if pc_for_cache is not None:
                            key = _cache_key(self.domain.platform_id, pc_for_cache)
                            with self._lock:
                                self._cache[key] = report
        else:
            # Fallback: per-item verify_one via thread pool.
            if not self._logged_no_batch:
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
         "facts": parsed.facts, "relations": parsed.relations,
         # Only add the key when a fan-in is present, so a facts-only completion
         # hashes byte-identically to the pre-#75 key (no cache churn).
         **({"predictions": parsed.predictions}
            if parsed.predictions is not None else {})},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
