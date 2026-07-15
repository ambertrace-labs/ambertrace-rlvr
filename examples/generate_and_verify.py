"""Generate decisions with a policy and verify each against the platform.

The inference counterpart to ``score_completions.py`` (which scores canned text):
this runs a live model end-to-end —

    prompt -> policy.generate -> parser -> AmberTrace verify -> reward

— so you can watch what a policy actually emits and whether the kernel certifies
it. Domain-agnostic: the platform, parser and reward shaper come from ``--config``;
the prompts come from a ``--prompts`` JSONL (the same chat-format the training/eval
sets use, e.g. produced by ``gen_training_prompts.py``). Nothing here is tied to a
particular domain. Only ``AMBERTRACE_API_KEY`` comes from the environment.

    # Grant Eligibility demo (base model vs. a trained checkpoint):
    python examples/generate_and_verify.py --prompts data/grant_eligibility_eval.jsonl
    python examples/generate_and_verify.py --prompts data/grant_eligibility_eval.jsonl \
        --model outputs/grant_eligibility_grpo/policy

    # Any other domain — point --config and --prompts at your own:
    python examples/generate_and_verify.py --config configs/your_run.yaml \
        --prompts data/your_eval.jsonl --model your-org/your-model

A base instruct model tends to emit off-schema facts the fact gate rejects (reward
near the floor); a policy trained against the certificate reward learns to emit
facts the kernel accepts. Point ``--model`` at a checkpoint to see the difference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ambertrace_rlvr import load_run_config
from ambertrace_rlvr.prompts import build_system_prompt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "grant_eligibility.yaml"

# Fields a --prompts row may carry a bare application string under (when it isn't
# already a full chat-message list). Checked in order.
_TEXT_KEYS = ("prompt", "application", "question", "text", "input")


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip())


def _to_messages(row: dict[str, Any], system_fallback: str) -> list[dict[str, str]] | None:
    """Coerce a prompts-file row into chat messages.

    Preferred form is the training/eval format: ``{"prompt": [ {role, content}, ... ]}``
    — used verbatim, so the domain's own system prompt travels with the data. As a
    fallback, a bare application string (under any of ``_TEXT_KEYS``) is wrapped with
    a generic system prompt built from the config's parser keys.
    """
    p = row.get("prompt")
    if isinstance(p, list) and p:
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in p]
    for key in _TEXT_KEYS:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return [{"role": "system", "content": system_fallback},
                    {"role": "user", "content": v}]
    return None


def _last_user(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return messages[-1]["content"] if messages else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--prompts", default=None,
                    help="JSONL of chat prompts (defaults to the config's eval/dataset path)")
    ap.add_argument("--model", default=None,
                    help="HF id or checkpoint dir (default: the config's training model)")
    ap.add_argument("--domain-name", default="the target",
                    help="domain label for the fallback system prompt (bare-string rows only)")
    ap.add_argument("-n", "--limit", type=int, default=3, help="number of prompts to run")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    _load_dotenv()

    run = load_run_config(args.config)
    parser = run.domain.parser

    prompts_path = args.prompts
    if prompts_path is None and run.eval is not None:
        prompts_path = run.eval.path
    if prompts_path is None and run.dataset is not None:
        prompts_path = run.dataset.path
    if not prompts_path:
        raise SystemExit("no --prompts given and the config declares no eval/dataset path")

    model_id = args.model or (run.training.model if run.training else None)
    if model_id is None:
        raise SystemExit("no --model given and the config has no [training] model")

    # Fallback system prompt for bare-string rows — matches the parser's key names.
    answer_key = getattr(parser, "answer_key", "classification")
    facts_key = getattr(parser, "facts_key", "facts")
    system_fallback = build_system_prompt(args.domain_name, answer_key=answer_key, facts_key=facts_key)

    with open(prompts_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"no prompts found in {prompts_path}")

    # MPS load workaround (mirrors examples/grant_eligibility_grpo.py).
    import transformers.core_model_loading as _cml  # type: ignore
    _cml.GLOBAL_WORKERS = 1
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {model_id} on {device} … (config: {args.config}, prompts: {prompts_path})",
          flush=True)
    tok: Any = AutoTokenizer.from_pretrained(model_id)
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = model.to(device)
    model.eval()

    def generate(messages: list[dict[str, str]]) -> str:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    shown = 0
    for row in rows:
        if shown >= args.limit:
            break
        messages = _to_messages(row, system_fallback)
        if messages is None:
            continue
        shown += 1
        application = _last_user(messages)
        print("\n" + "=" * 78 + f"\nPROMPT {shown}: {application}")
        completion = generate(messages)
        print("\n--- MODEL COMPLETION ---\n" + completion.strip())

        parsed = parser.parse(application, completion)
        if parsed is None:
            print("\n--- VERDICT --- unparseable (no <decision> block) → reward floors")
            continue
        report = run.verifier.verify_one(parsed)
        reward = run.shaper.score(parsed, report, None).total
        print("\n--- VERDICT ---")
        print(f"  model says     : {parsed.proposed_answer!r}  facts={parsed.facts}")
        print(f"  platform       : decision={report.raw.get('decision')!r} "
              f"proof_checked={report.proof_checked}")
        print(f"  facts accepted : {report.fact_summary.get('accepted', 0)} "
              f"rejected={report.fact_summary.get('rejected', len(report.rejected_facts))}")
        print(f"  REWARD         : {reward:+.3f}")

    if shown == 0:
        raise SystemExit(f"no usable prompts in {prompts_path} "
                         f"(expected a 'prompt' chat list or a text field: {_TEXT_KEYS})")


if __name__ == "__main__":
    main()
