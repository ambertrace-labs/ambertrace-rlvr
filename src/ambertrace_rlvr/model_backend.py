"""Model backends for the evaluation lane: call a model under test and return its
raw completion, with the network fully injectable so tests never touch it.

The primary backend is **LM Studio**'s OpenAI-compatible local server
(``http://localhost:1234/v1``), which lets us evaluate the *actual open weights at
a known quantization* on this machine — not a hosted provider's served config. A
backend only produces text; coercing that text into a domain's label space lives
in :func:`ambertrace_rlvr.deviation.parse_model_answer`, and the eval flows take a
plain ``prompt -> completion`` callable (see :meth:`LMStudioProvider.as_model`).

No third-party HTTP dependency: the default transport is a stdlib POST. Tests
inject their own transport, so importing this module — or the package — never
opens a socket.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# A transport maps (url, payload, timeout) -> the parsed response dict (OpenAI
# chat-completions shape). Injectable so tests are network-free.
Transport = Callable[[str, "dict[str, Any]", float], "dict[str, Any]"]


class ModelBackendError(RuntimeError):
    """A backend could not produce a completion (connection / HTTP / shape error).
    Distinct from an *empty* completion, which is a model refusal, not an error."""


@runtime_checkable
class ModelProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


@dataclass(frozen=True)
class ScoredDecision:
    """A typed contestant's answer to one decision item: the chosen label plus the
    probability mass it placed on each candidate and an overall confidence.

    Unlike a chat completion (raw text coerced later), a *System-1* model — Jev,
    Nimble — returns the label and its calibrated probabilities directly. Capturing
    them here is what lets the eval score **calibration** (Brier / ECE), not just
    accuracy. ``choice is None`` is a refusal / non-answer (fail-closed), never a
    silent wrong label; ``probabilities`` is empty for backends that expose only a
    label (e.g. a chat model wrapped as a degenerate decider)."""

    choice: str | None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    @property
    def answered(self) -> bool:
        return self.choice is not None


@runtime_checkable
class TypedDecider(Protocol):
    """A contestant that decides over a *known candidate set*. The candidates are
    needed at request time (a Choice question, not free text), so — unlike the
    chat ``Model = Callable[[str], str]`` seam — the label space is passed in."""

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision: ...


@dataclass
class LMStudioProvider:
    """Call a model served by LM Studio's OpenAI-compatible endpoint.

    ``temperature`` defaults to 0.0 (pinned for reproducible eval runs). A
    connection/HTTP failure raises :class:`ModelBackendError` (so a whole run
    against a down server is surfaced, not silently scored as all-refusals); a
    well-formed-but-empty/truncated response returns ``""`` — a refusal downstream,
    never a wrong label."""

    model: str
    base_url: str = "http://localhost:1234/v1"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    system: str | None = None
    # Extra fields merged into the request payload — e.g. reasoning controls for
    # thinking models: ``{"chat_template_kwargs": {"enable_thinking": False}}`` or
    # ``{"reasoning_effort": "low"}``, so a reasoning model answers the decision
    # directly instead of spending its whole token budget on <think> and
    # truncating into a false refusal.
    extra_body: dict[str, Any] | None = None
    # Injectable for tests: defaults to a stdlib HTTP POST (no third-party dep).
    transport: Transport | None = None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        sys_prompt = system if system is not None else self.system
        data = self._chat(sys_prompt, prompt)
        # Some model templates (e.g. Mistral v0.3) accept only user/assistant
        # roles and error on a system message. Fold the system prompt into the
        # user turn and retry, so the matrix runs uniformly across families.
        if sys_prompt and _is_role_error(data):
            data = self._chat(None, f"{sys_prompt}\n\n{prompt}")
        return _extract_content(data)

    def _chat(self, sys_prompt: str | None, user: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            payload.update(self.extra_body)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        transport = self.transport or _http_post
        try:
            return transport(url, payload, self.timeout)
        except ModelBackendError:
            raise
        except Exception as e:  # a foreign transport failure — normalise it
            raise ModelBackendError(f"model request failed: {e!r}") from e

    def complete_full(
        self, prompt: str, *, system: str | None = None,
    ) -> tuple[str, str, str]:
        """Like :meth:`complete` but returns ``(content, finish_reason,
        reasoning_content)``.

        ``reasoning_content`` is the thinking trace from models that surface it
        as a separate API field (e.g. Qwen3 in LM Studio), or ``""`` when
        absent. ``finish_reason`` is ``"stop"`` / ``"length"`` / ``""``."""
        sys_prompt = system if system is not None else self.system
        data = self._chat(sys_prompt, prompt)
        if sys_prompt and _is_role_error(data):
            data = self._chat(None, f"{sys_prompt}\n\n{prompt}")
        return (
            _extract_content(data),
            _extract_finish_reason(data),
            _extract_reasoning_content(data),
        )

    def as_model(self) -> Callable[[str], str]:
        """Adapt to the ``prompt -> completion`` callable the eval sweeps expect."""
        return lambda prompt: self.complete(prompt)


@dataclass
class JevProvider:
    """Call TypeSafe's **Jev** System-One model as a :class:`TypedDecider`.

    Jev answers a typed *Choice* question — pick one of a supplied candidate set —
    and returns the chosen label plus a probability for every candidate and a
    confidence, in one non-autoregressive step (no generated text). The eval maps
    a decision item's ``vocabulary`` onto the Choice ``criteria`` and reads the
    answer straight back, so Jev is scored on the same corpus as every other
    contestant (see :func:`ambertrace_rlvr.typed_decision.run_typed_model`).

    API: ``POST {base_url}/systemone`` with ``Authorization: Bearer <key>``. The
    key defaults to the ``TYPESAFE_API_KEY`` env var; never hard-code it. A
    connection/HTTP failure raises :class:`ModelBackendError` (a down API is
    surfaced, not scored as all-refusals); a well-formed response that carries no
    usable choice yields a refusal ``ScoredDecision(choice=None)``. ``transport``
    is injectable so tests never open a socket."""

    api_key: str | None = None
    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai/v1"
    instructions: str = "Choose exactly one action for this case."
    timeout: float = 30.0
    transport: Transport | None = None

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision:
        criteria = {verb: verb for verb in label_space}
        payload: dict[str, Any] = {
            "state": prompt,
            "model": self.model,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": criteria,
                }
            },
        }
        url = f"{self.base_url.rstrip('/')}/systemone"
        transport = self.transport or _authed_post(self._resolve_key())
        try:
            data = transport(url, payload, self.timeout)
        except ModelBackendError:
            raise
        except Exception as e:  # a foreign transport failure — normalise it
            raise ModelBackendError(f"Jev request failed: {e!r}") from e
        return _parse_jev_choice(data, label_space)

    def _resolve_key(self) -> str:
        key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ModelBackendError(
                "no Jev API key: pass api_key= or set TYPESAFE_API_KEY")
        return key


def _authed_post(api_key: str) -> Transport:
    """A :data:`Transport` that POSTs JSON with a Bearer token (Jev auth)."""
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return _http_post(url, payload, timeout, headers=headers)

    return post


def _parse_jev_choice(
    data: dict[str, Any], label_space: Sequence[str]
) -> ScoredDecision:
    """Read the ``action`` answer from a Jev response into a :class:`ScoredDecision`.

    Tolerant of shape drift: a missing/odd field yields a refusal (choice=None),
    never a crash. A returned choice outside ``label_space`` is dropped to a
    refusal rather than scored as a (wrong) label."""
    answer = data.get("answers", {}).get("action") if isinstance(data, dict) else None
    if not isinstance(answer, dict):
        return ScoredDecision(choice=None)
    raw_probs = answer.get("probabilities")
    probs: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            if k in label_space and isinstance(v, (int, float)):
                probs[str(k)] = float(v)
    choice = answer.get("choice")
    choice = choice if isinstance(choice, str) and choice in label_space else None
    conf = answer.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) else None
    return ScoredDecision(choice=choice, probabilities=probs, confidence=confidence)


@runtime_checkable
class NimbleScorer(Protocol):
    """The local Nimble scorer surface we depend on (``nimble.scoring``'s
    ``ParallelScorer`` / ``CudaCandidateScorer``): score ``text`` against a flat
    enum/boolean ``schema`` and return the typed decisions plus per-candidate
    scores. Kept as a Protocol so tests inject a fake and the ``nimble`` package is
    never a hard dependency of this repo."""

    def score(self, text: str, schema: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class NimbleProvider:
    """Call Bespoke's **Nimble** (`Bespoke-Nimble-9B`) as a :class:`TypedDecider`.

    Nimble reads logits over the candidate answer tokens and returns the chosen
    label plus a probability for each — in one local, non-autoregressive step. The
    eval maps a decision item's ``vocabulary`` onto a single-field ``enum`` schema
    and reads the answer straight back, so Nimble is scored on the same corpus as
    every other contestant.

    ``scorer`` is the loaded Nimble scorer (see :meth:`load`); it is injected so
    tests never touch the model. A scorer failure raises :class:`ModelBackendError`
    (a broken runtime is surfaced, not scored as all-refusals); a response with no
    usable choice is a refusal ``ScoredDecision(choice=None)``."""

    scorer: NimbleScorer
    field: str = "action"
    description: str = "The single action to take for this case."

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision:
        schema = {
            self.field: {
                "type": "enum",
                "choices": list(label_space),
                "description": self.description,
            }
        }
        try:
            result = self.scorer.score(prompt, schema)
        except Exception as e:  # a broken local runtime — surface it, don't hide it
            raise ModelBackendError(f"Nimble scorer failed: {e!r}") from e
        return _parse_nimble_field(result, self.field, label_space)

    @classmethod
    def load(
        cls, config_path: str = ".cache/nimble-model.json", *, cuda: bool = False,
        field: str = "action",
    ) -> NimbleProvider:
        """Build a provider from a local Nimble model config, lazily importing
        ``nimble`` so the package is only required for a live run."""
        import json as _json
        from pathlib import Path as _Path

        config = _json.loads(_Path(config_path).read_text())
        if cuda:
            from nimble.scoring.cuda_scorer import (  # type: ignore[import-untyped]
                CudaCandidateScorer,
            )
            scorer: NimbleScorer = CudaCandidateScorer(**config)
        else:
            from nimble.scoring.parallel_scorer import (  # type: ignore[import-untyped]
                ParallelScorer,
            )
            scorer = ParallelScorer(**config)
        return cls(scorer=scorer, field=field)


def _parse_nimble_field(
    result: dict[str, Any], field: str, label_space: Sequence[str]
) -> ScoredDecision:
    """Read one enum field from a Nimble ``score`` result into a
    :class:`ScoredDecision`. Tolerant of shape (``output`` map or a flat map for
    the choice; ``fields[f].scores`` for probabilities); a choice outside
    ``label_space`` drops to a refusal rather than a wrong label."""
    if not isinstance(result, dict):
        return ScoredDecision(choice=None)
    output = result.get("output")
    holder = output if isinstance(output, dict) else result
    choice = holder.get(field)
    choice = choice if isinstance(choice, str) and choice in label_space else None
    probs: dict[str, float] = {}
    fields = result.get("fields")
    if isinstance(fields, dict) and isinstance(fields.get(field), dict):
        raw_scores = fields[field].get("scores")
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                if k in label_space and isinstance(v, (int, float)):
                    probs[str(k)] = float(v)
    confidence = probs.get(choice) if choice is not None else None
    return ScoredDecision(choice=choice, probabilities=probs, confidence=confidence)


def _http_post(
    url: str, payload: dict[str, Any], timeout: float,
    *, headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers=headers or {"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # A 4xx/5xx with a body is a *response* (e.g. a template role error the
        # caller can recover from), not an unreachable server — surface the body.
        try:
            raw = e.read().decode("utf-8")
        except (OSError, ValueError):
            raise ModelBackendError(f"HTTP {e.code} from {url}: {e!r}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ModelBackendError(f"cannot reach model server at {url}: {e!r}") from e
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise ModelBackendError(f"invalid JSON from {url}: {e!r}") from e
    if not isinstance(parsed, dict):
        raise ModelBackendError(f"unexpected response shape from {url}")
    return parsed


def _is_role_error(data: dict[str, Any]) -> bool:
    """Whether the response is a template error about unsupported message roles
    (i.e. the model accepts only user/assistant, not a system message)."""
    err = data.get("error")
    text = (err if isinstance(err, str) else str(err)) if err is not None else ""
    low = text.lower()
    return "role" in low and ("system" in low or "user and assistant" in low)


def _extract_content(data: dict[str, Any]) -> str:
    """Pull the assistant text from an OpenAI chat-completions response. Returns
    ``""`` (a downstream refusal) rather than raising on a missing/odd field, so a
    truncated or empty generation is a non-answer, not a crash."""
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_finish_reason(data: dict[str, Any]) -> str:
    """Pull ``finish_reason`` from the first choice. Returns ``""`` when absent."""
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_reasoning_content(data: dict[str, Any]) -> str:
    """Pull ``reasoning_content`` from the assistant message (the thinking
    trace from models that surface it as a separate field, e.g. Qwen3 in
    LM Studio). Returns ``""`` when absent."""
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        rc = message.get("reasoning_content")
        return rc if isinstance(rc, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""
