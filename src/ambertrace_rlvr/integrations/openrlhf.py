"""OpenRLHF adapter — a remote reward-model-server shim over HTTP.

OpenRLHF can drive reward from a *remote reward model*: rollout workers POST a
batch of decoded rollouts to an HTTP endpoint (default path ``/get_reward``) and
read scalar rewards back. The observed contract (OpenRLHF ``SingleTurnAgentExecutor``
and its reference ``serve_rm.py``) is::

    POST /get_reward
    {"query": [str, ...], "prompts": [str, ...], "labels": [any, ...]}
    -> {"rewards": [float, ...], "scores": [float, ...]}

``query`` items are the decoded prompt+response strings; the client reads
``scores`` (falling back to ``rewards``). This module wraps our batched reward
function (``reward_fn(prompts, completions, metadata) -> list[float]``) behind
that contract as a dependency-free WSGI app — no RL-algorithm logic here, in
parity with the TRL/veRL adapters.

Design invariants:

* **Fail-closed** — any error scoring a batch resolves to the reward floor for
  every item and is logged; the server never raises into the client. Malformed
  requests get a 4xx (the batch size is unknown, so no reward can be returned).
* **Scoped key, never logged** — if an API key is configured the app requires it
  on every request (``Authorization: Bearer <key>`` or ``X-API-Key: <key>``,
  constant-time compared). Keys are never written to logs or responses.
* **Read-only** — the runtime only queries the platform through the verifier; the
  app adds no authoring or platform-write surface.

OpenRLHF's stock HTTP client sends no auth header, so to use ``api_key`` either
front the server with a proxy that injects the header or run it key-less on a
trusted network. See ``examples/openrlhf_reward_server.py`` for client wiring.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from ..verifier import RewardFunction

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "/get_reward"
API_KEY_ENV = "AMBERTRACE_RLVR_REWARD_KEY"

# Minimal WSGI types (stdlib only; no framework dependency).
StartResponse = Callable[[str, list[tuple[str, str]]], Any]
WSGIApp = Callable[[dict[str, Any], StartResponse], list[bytes]]


def score_batch(
    reward_fn: RewardFunction,
    queries: Sequence[Any],
    prompts: Sequence[Any] | None = None,
    labels: Sequence[Any] | None = None,
    *,
    floor: float = -1.0,
    gold_key: str = "gold",
) -> list[float]:
    """Map an OpenRLHF request batch to a list of rewards, fail-closed.

    ``queries`` are the decoded prompt+response strings OpenRLHF scores; each is
    passed through as a completion (our parser scans it for the decision block).
    ``prompts`` are attached as the per-sample prompt when present; ``labels`` are
    forwarded as ``metadata[gold_key]`` — the optional correctness signal the
    shaper reads. Any exception floors the whole batch and is logged.
    """
    completions = [_flatten(q) for q in queries]
    n = len(completions)
    prompt_seq = _align(prompts, n)
    metadata: list[dict[str, Any]] = [{} for _ in range(n)]
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        for i, label in enumerate(labels):
            if i < n and label is not None:
                metadata[i][gold_key] = label
    try:
        rewards = reward_fn(prompt_seq, completions, metadata)
        return [float(r) for r in rewards]
    except Exception:  # fail-closed: never raise into the training loop
        logger.exception("openrlhf reward batch failed; flooring %d item(s)", n)
        return [floor] * n


def build_openrlhf_reward_app(
    reward_fn: RewardFunction,
    *,
    api_key: str | None = None,
    floor: float = -1.0,
    endpoint: str = DEFAULT_ENDPOINT,
    gold_key: str = "gold",
) -> WSGIApp:
    """Build a stdlib WSGI app exposing ``reward_fn`` over OpenRLHF's remote-RM
    HTTP contract.

    ``api_key`` (or the ``AMBERTRACE_RLVR_REWARD_KEY`` env var) enables auth; when
    unset the app runs key-less. ``endpoint`` is the accepted POST path. The
    returned callable is a plain WSGI app — offline tests can call it in-process
    (no socket); :func:`serve_openrlhf_reward` runs it on a real port.
    """
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)

    def app(environ: dict[str, Any], start_response: StartResponse) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "")
        path = environ.get("PATH_INFO", "") or "/"
        if method != "POST":
            return _respond(start_response, "405 Method Not Allowed",
                            {"error": "POST required"})
        if endpoint and path != endpoint:
            return _respond(start_response, "404 Not Found",
                            {"error": "unknown endpoint"})
        if not _authorized(environ, key):
            # Never log the key or the presented credential — only the outcome.
            logger.warning("openrlhf reward request rejected: missing/invalid API key")
            return _respond(start_response, "401 Unauthorized",
                            {"error": "unauthorized"})
        try:
            data = _read_json(environ)
            queries = data.get("query")
            if queries is None:
                queries = data.get("queries")
            if not isinstance(queries, list):
                raise ValueError("request must contain a 'query' list")
        except Exception:
            # Batch size is unknown, so there is nothing to floor — reject.
            logger.warning("openrlhf reward request rejected: malformed request body")
            return _respond(start_response, "400 Bad Request",
                            {"error": "malformed request"})
        rewards = score_batch(
            reward_fn, queries, data.get("prompts"), data.get("labels"),
            floor=floor, gold_key=gold_key,
        )
        # Mirror the reference server: return both keys the client may read.
        return _respond(start_response, "200 OK", {"rewards": rewards, "scores": rewards})

    return app


def serve_openrlhf_reward(
    reward_fn: RewardFunction,
    *,
    host: str = "0.0.0.0",
    port: int = 5000,
    api_key: str | None = None,
    floor: float = -1.0,
    endpoint: str = DEFAULT_ENDPOINT,
) -> None:  # pragma: no cover - binds a real socket; exercised via the app in tests
    """Serve the reward app on ``host:port`` with the stdlib WSGI server (blocking).

    Point OpenRLHF at it with ``--remote_rm_url http://<host>:<port>{endpoint}``.
    """
    from wsgiref.simple_server import make_server

    app = build_openrlhf_reward_app(
        reward_fn, api_key=api_key, floor=floor, endpoint=endpoint,
    )
    with make_server(host, port, app) as httpd:
        logger.info("serving AmberTrace OpenRLHF reward app on %s:%d%s (auth=%s)",
                    host, port, endpoint,
                    "on" if (api_key or os.environ.get(API_KEY_ENV)) else "off")
        httpd.serve_forever()


def _authorized(environ: dict[str, Any], key: str | None) -> bool:
    if not key:  # no key configured -> auth disabled
        return True
    presented = ""
    header = environ.get("HTTP_AUTHORIZATION", "")
    if header.startswith("Bearer "):
        presented = header[len("Bearer "):].strip()
    if not presented:
        presented = environ.get("HTTP_X_API_KEY", "")
    return bool(presented) and hmac.compare_digest(presented, key)


def _read_json(environ: dict[str, Any]) -> dict[str, Any]:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        length = 0
    body = environ["wsgi.input"].read(length) if length > 0 else b""
    data = json.loads(body) if body else {}
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _respond(start_response: StartResponse, status: str,
             payload: dict[str, Any]) -> list[bytes]:
    body = json.dumps(payload).encode("utf-8")
    start_response(status, [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ])
    return [body]


def _align(prompts: Sequence[Any] | None, n: int) -> list[str]:
    if isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
        return [str(prompts[i]) if i < len(prompts) and prompts[i] is not None else ""
                for i in range(n)]
    return [""] * n


def _flatten(completion: Any) -> str:
    # Tolerate the conversational format [{"role": ..., "content": ...}, ...].
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)
