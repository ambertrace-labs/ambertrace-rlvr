"""Serve an AmberTrace verified reward to OpenRLHF over HTTP — the remote
reward-model counterpart to the TRL/veRL examples.

The reward is identical to every other example: AmberTrace's proof certificate,
turned into a scalar by `DefaultRewardShaper`. Only the *plumbing* differs —
OpenRLHF drives reward from a remote HTTP reward model: rollout workers POST a
batch of decoded rollouts and read scalar rewards back. This exposes our batched
reward function behind that contract as a dependency-free WSGI app (stdlib only).

    # Offline: exercise the app in-process — no socket, no network, FakeVerifier.
    python examples/openrlhf_reward_server.py --dry-run

    # Live: serve the real verifier (needs the platform reachable + AMBERTRACE_API_KEY).
    python examples/openrlhf_reward_server.py --host 0.0.0.0 --port 5000
    # optionally require a scoped key on every request:
    AMBERTRACE_RLVR_REWARD_KEY=my-scoped-key python examples/openrlhf_reward_server.py

Then point OpenRLHF at it:

    openrlhf.cli.train_ppo_ray ... --remote_rm_url http://<host>:5000/get_reward

Contract (observed from OpenRLHF's SingleTurnAgentExecutor / serve_rm.py):

    POST /get_reward
    {"query": [str, ...], "prompts": [str, ...], "labels": [any, ...]}
    -> {"rewards": [float, ...], "scores": [float, ...]}

`query` items are decoded prompt+response strings; `labels` (if sent) are the
gold signal, forwarded to the shaper as per-sample metadata. The app is
fail-closed: any scoring error floors the batch and is logged, never raised to
the trainer.

Auth note: OpenRLHF's stock HTTP client sends no auth header. When
AMBERTRACE_RLVR_REWARD_KEY is set the app requires `Authorization: Bearer <key>`
(or `X-API-Key: <key>`) on every request, so run key-less on a trusted network or
front the server with a proxy that injects the header.

Config: configs/grant_eligibility.yaml (reuses the demo platform).
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from ambertrace_rlvr import load_run_config
from ambertrace_rlvr.integrations.openrlhf import (
    DEFAULT_ENDPOINT,
    build_openrlhf_reward_app,
    serve_openrlhf_reward,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "grant_eligibility.yaml"

# A well-formed sample rollout (prompt+response) for the offline dry-run.
_SAMPLE_QUERY = (
    "Assess this applicant for the grant.\n"
    "<reasoning>Adult, low income, resident, no active grant — all criteria met.</reasoning>"
    '<decision>{"classification": "permit", "facts": {"age": 40, "annual_income": 25000, '
    '"resident": true, "has_active_grant": false}}</decision>'
)
_MALFORMED_QUERY = "I could not determine a decision for this applicant."


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _post_in_process(app: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the WSGI app directly (no socket) — the same path OpenRLHF's HTTP
    client drives, minus the network."""
    raw = json.dumps(payload).encode("utf-8")
    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": DEFAULT_ENDPOINT,
        "CONTENT_LENGTH": str(len(raw)),
        "wsgi.input": BytesIO(raw),
    }
    status_holder: dict[str, str] = {}

    def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
        status_holder["status"] = status

    chunks = app(environ, start_response)
    return json.loads(b"".join(chunks).decode("utf-8"))


def dry_run() -> None:
    """Exercise the OpenRLHF reward app offline — no server, no GPU, no network."""
    from ambertrace_rlvr.testing import FakeVerifier

    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function())
    result = _post_in_process(app, {"query": [_SAMPLE_QUERY, _MALFORMED_QUERY]})
    good, bad = result["rewards"]
    print(f"dry-run /get_reward: well-formed={good:.3f}  malformed={bad:.3f}")
    assert good > bad, "well-formed permit should out-score malformed floor"
    print("OK — OpenRLHF reward app is sound (well-formed > malformed floor).")


def serve(host: str, port: int) -> None:
    """Serve the live verifier as an OpenRLHF remote reward model."""
    _load_dotenv()
    run = load_run_config(CONFIG)
    reward_fn = run.reward_function()
    auth = "on" if os.environ.get("AMBERTRACE_RLVR_REWARD_KEY") else "off"
    print(f"serving AmberTrace reward (platform {run.domain.platform_id}) for OpenRLHF "
          f"at http://{host}:{port}{DEFAULT_ENDPOINT} (auth={auth})")
    print("point OpenRLHF at it with: "
          f"--remote_rm_url http://<host>:{port}{DEFAULT_ENDPOINT}")
    serve_openrlhf_reward(reward_fn, host=host, port=port)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="offline in-process check (no server/GPU/network)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
