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
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
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
    # Injectable for tests: defaults to a stdlib HTTP POST (no third-party dep).
    transport: Transport | None = None

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        sys_prompt = system if system is not None else self.system
        messages: list[dict[str, str]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        transport = self.transport or _http_post
        try:
            data = transport(url, payload, self.timeout)
        except ModelBackendError:
            raise
        except Exception as e:  # a foreign transport failure — normalise it
            raise ModelBackendError(f"model request failed: {e!r}") from e
        return _extract_content(data)

    def as_model(self) -> Callable[[str], str]:
        """Adapt to the ``prompt -> completion`` callable the eval sweeps expect."""
        return lambda prompt: self.complete(prompt)


def _http_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed scheme)
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as e:
        raise ModelBackendError(f"cannot reach model server at {url}: {e!r}") from e
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise ModelBackendError(f"invalid JSON from {url}: {e!r}") from e
    if not isinstance(parsed, dict):
        raise ModelBackendError(f"unexpected response shape from {url}")
    return parsed


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
