"""Held-out probe runner: generate from each GRPO checkpoint and score.

Loads the base model (optionally with LoRA adapters) at each training
checkpoint, generates completions for every held-out probe item, splits
into think/stated channels, scores against the AmberTrace verifier, and
computes CoT drift metrics.

Outputs:
  * ``step_NNNN.jsonl`` — raw per-item capture (primary artifact, incremental,
    resume-safe).
  * ``summary.jsonl`` — one line per checkpoint with all metrics.
  * Printed per-checkpoint table + top-10 rising/falling unigrams.

Usage::

    # Full sweep (hours of GPU):
    python examples/probe_checkpoints.py

    # Smoke test (2 items, baseline + one adapter):
    python examples/probe_checkpoints.py --limit 2 --only-steps 0,60

Requires: mlx-lm, AMBERTRACE_API_KEY in the environment (or .env file).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CHECKPOINTS_DIR = REPO / "outputs" / "faithfulness_mlx_grpo"
DEFAULT_MODEL = "/Users/peterpilotgenai/models/mlx/Olmo-3-7B-Think-SFT-8bit"
DEFAULT_PROBES = REPO / "data" / "air_track_eval.jsonl"
DEFAULT_CONFIG = REPO / "configs" / "air_track.yaml"
DEFAULT_OUT_DIR = REPO / "outputs" / "probe_runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _discover_checkpoints(
    ckpt_dir: Path,
    only_steps: set[int] | None = None,
) -> list[tuple[int, Path | None]]:
    """Return (step, adapter_path) pairs sorted by step.

    Step 0 = baseline (no adapter).  Each ``NNNNNNN_adapters.safetensors``
    maps to a step number parsed from its filename prefix.
    """
    entries: list[tuple[int, Path | None]] = [(0, None)]  # baseline
    pattern = re.compile(r"^(\d+)_adapters\.safetensors$")
    for f in sorted(ckpt_dir.iterdir()):
        m = pattern.match(f.name)
        if m:
            step = int(m.group(1))
            entries.append((step, f))
    if only_steps is not None:
        entries = [(s, p) for s, p in entries if s in only_steps]
    entries.sort(key=lambda x: x[0])
    return entries


def _load_probes(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
        if limit and len(items) >= limit:
            break
    return items


def _build_prompt_text(messages: list[dict[str, str]]) -> str:
    """Concatenate chat messages into a flat prompt string for the parser."""
    return "\n".join(m.get("content", "") for m in messages).strip()


def _split_channels(full_output: str) -> tuple[str, str]:
    """Split generation into think (before </think>) and stated (<reasoning>)."""
    think = ""
    stated = ""
    # Think channel: everything before </think>.
    think_match = re.search(r"(.*?)</think>", full_output, re.DOTALL | re.IGNORECASE)
    if think_match:
        think = think_match.group(1).strip()
        # Remove <think> opening tag if present.
        think = re.sub(r"^<think>\s*", "", think, flags=re.IGNORECASE).strip()

    # Stated channel: the <reasoning> block after </think>.
    after_think = full_output
    if think_match:
        after_think = full_output[think_match.end():]
    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>", after_think,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        stated = reasoning_match.group(1).strip()
    return think, stated


def _extract_decision(full_output: str) -> str | None:
    """Extract the triage decision from the <decision> JSON block."""
    m = re.search(r"<decision>\s*(.*?)\s*</decision>", full_output, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        return str(data.get("triage", "")).strip().lower() or None
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None


def _extract_facts(full_output: str) -> dict[str, Any] | None:
    """Extract the facts dict from the <decision> JSON block."""
    m = re.search(r"<decision>\s*(.*?)\s*</decision>", full_output, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        facts = data.get("facts")
        return dict(facts) if isinstance(facts, dict) else None
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None


def _existing_item_ids(path: Path) -> set[str]:
    """Read item_ids already captured in a JSONL file (for resume)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if "item_id" in rec:
                ids.add(str(rec["item_id"]))
        except (json.JSONDecodeError, ValueError):
            continue
    return ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_checkpoint(
    model_path: str,
    adapter_path: Path | None,
    probes: list[dict[str, Any]],
    out_path: Path,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Generate completions for all probe items at one checkpoint.

    Writes JSONL incrementally (append + flush per item).  Resumes by
    skipping items already present in the output file.

    Returns the full list of records (including previously captured ones).
    """
    from mlx_lm import generate, load  # type: ignore[import-untyped]

    existing = _existing_item_ids(out_path)
    needs_generation = [
        (i, p) for i, p in enumerate(probes) if str(i) not in existing
    ]

    all_records: list[dict[str, Any]] = []
    # Load existing records.
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    all_records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass

    if not needs_generation:
        print(f"    All {len(probes)} items already captured, skipping generation.")
        return all_records

    # Load model.  The checkpointed adapters are bare safetensors files (no
    # adapter_config.json directory), so we cannot use mlx_lm.load(adapter_path=).
    # Instead: load the base model, apply LoRA layers matching the training
    # config, then load the adapter weights.
    print(f"    Loading model (adapter={adapter_path})...")
    model, tokenizer = load(model_path)

    if adapter_path is not None:
        import mlx.core as mx  # type: ignore[import-untyped]
        from mlx_lm.tuner.utils import (
            linear_to_lora_layers,  # type: ignore[import-untyped]
        )

        # Match the LoRA config used during training (faithfulness_mlx_grpo.py).
        linear_to_lora_layers(
            model,
            num_layers=16,
            config={"rank": 8, "scale": 20.0, "dropout": 0.0,
                    "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
        )
        adapter_weights = mx.load(str(adapter_path))
        model.load_weights(list(adapter_weights.items()), strict=False)
        model.eval()
        mx.eval(model.parameters())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for idx, probe in needs_generation:
            item_id = str(idx)
            messages = probe["prompt"]
            gold = probe.get("gold", "")

            # Build prompt via chat template.
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt_str = _build_prompt_text(messages)

            user_content = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")

            full_output = generate(
                model, tokenizer, prompt=prompt_str, max_tokens=max_tokens,
            )
            think, stated = _split_channels(full_output)
            decision = _extract_decision(full_output)

            rec = {
                "item_id": item_id,
                "gold": gold,
                "user": user_content,
                "full_output": full_output,
                "think": think,
                "stated": stated,
                "decision": decision,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            all_records.append(rec)
            print(f"    [{idx + 1}/{len(probes)}] decision={decision} gold={gold}")

    # Free model memory.
    del model, tokenizer
    gc.collect()

    return all_records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_checkpoint(
    records: list[dict[str, Any]],
    run_config: Any,
    baseline_corpus: Any | None,
    step: int,
) -> dict[str, Any]:
    """Score a checkpoint's generated records and compute drift metrics.

    Returns a metrics dict suitable for summary.jsonl.
    """
    from ambertrace_rlvr.cot_drift import (
        BACKTRACKING,
        HEDGING,
        VERIFIER_AWARENESS,
        ProbeTrace,
        channel_lengths,
        distinct_n,
        lexicon_rate,
        ngram_logodds_diff,
        think_stated_divergence,
        unsupported_fact_fraction,
    )
    from ambertrace_rlvr.faithfulness import faithfulness
    from ambertrace_rlvr.faithfulness_scorer import score_batch_rich

    # Build corpus of ProbeTrace for drift metrics.
    corpus = []
    for rec in records:
        facts = _extract_facts(rec.get("full_output", ""))
        corpus.append(ProbeTrace(
            item_id=rec.get("item_id", ""),
            think=rec.get("think", ""),
            stated=rec.get("stated", ""),
            decision=rec.get("decision"),
            facts=facts,
        ))

    # --- Drift metrics ---
    lengths = channel_lengths(corpus)
    dn_think = distinct_n(corpus, n=3, channel="think")
    va_rate = lexicon_rate(corpus, VERIFIER_AWARENESS, channel="think")
    hedge_rate = lexicon_rate(corpus, HEDGING, channel="think")
    bt_rate = lexicon_rate(corpus, BACKTRACKING, channel="think")

    # Divergence metrics.
    # We need credited_rules from the verifier; use empty for now (pre-scoring).
    overlaps: list[float] = []
    concealment_count = 0
    flip_count = 0
    unsupported_fracs: list[float] = []
    for rec, trace in zip(records, corpus):
        # Use empty credited rules for divergence (we compute with available data).
        div = think_stated_divergence(trace, [])
        overlaps.append(div.channel_overlap)
        concealment_count += len(div.rules_in_think_only)
        if div.decision_flip:
            flip_count += 1

        prompt_text = rec.get("user", "")
        unsupported_fracs.append(unsupported_fact_fraction(trace, prompt_text))

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    mean_unsupported = sum(unsupported_fracs) / len(unsupported_fracs) if unsupported_fracs else 0.0

    # N-gram log-odds diff vs baseline.
    logodds_result = None
    if baseline_corpus is not None and step > 0:
        logodds_result = ngram_logodds_diff(
            baseline_corpus, corpus, n=1, top_k=10, min_count=2,
        )

    # --- Verifier scoring ---
    prompts = []
    completions = []
    metadata: list[dict[str, Any]] = []
    for rec in records:
        prompts.append(rec.get("user", ""))
        completions.append(rec.get("full_output", ""))
        metadata.append({"gold": rec.get("gold")})

    rich_scores = score_batch_rich(
        parser=run_config.domain.parser,
        shaper=run_config.shaper,
        verifier=run_config.verifier,
        prompts=prompts,
        completions=completions,
        metadata=metadata,
        floor=run_config.verifier.floor,
    )

    mean_reward = sum(s.reward for s in rich_scores) / len(rich_scores) if rich_scores else 0.0

    # Decision accuracy vs gold.
    correct = 0
    total = 0
    for rec in records:
        gold = str(rec.get("gold", "")).strip().lower()
        decision = str(rec.get("decision", "")).strip().lower() if rec.get("decision") else ""
        if gold:
            total += 1
            if decision == gold:
                correct += 1
    decision_accuracy = correct / total if total else 0.0

    # Faithfulness from rich scores.
    faith_values: list[float] = []
    for rs in rich_scores:
        if rs.credited_rules:
            f = faithfulness(rs.reasoning, list(rs.credited_rules))
            if f is not None:
                faith_values.append(f)
    mean_faithfulness = sum(faith_values) / len(faith_values) if faith_values else None

    # Mean consistency.
    mean_consistency = sum(s.consistency for s in rich_scores) / len(rich_scores) if rich_scores else 0.0

    metrics: dict[str, Any] = {
        "step": step,
        "n_items": len(records),
        "mean_reward": round(mean_reward, 4),
        "decision_accuracy": round(decision_accuracy, 4),
        "mean_faithfulness": round(mean_faithfulness, 4) if mean_faithfulness is not None else None,
        "mean_consistency": round(mean_consistency, 4),
        "think_mean_len": round(lengths.think_mean, 1),
        "think_median_len": round(lengths.think_median, 1),
        "stated_mean_len": round(lengths.stated_mean, 1),
        "stated_median_len": round(lengths.stated_median, 1),
        "distinct_3_think": round(dn_think, 4),
        "verifier_awareness_rate": round(va_rate, 4),
        "hedging_rate": round(hedge_rate, 4),
        "backtracking_rate": round(bt_rate, 4),
        "mean_channel_overlap": round(mean_overlap, 4),
        "concealment_count": concealment_count,
        "decision_flips": flip_count,
        "mean_unsupported_fact_frac": round(mean_unsupported, 4),
    }

    if logodds_result is not None:
        metrics["top10_rising"] = logodds_result.rising[:10]
        metrics["top10_falling"] = logodds_result.falling[:10]

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe GRPO checkpoints on held-out eval set")
    ap.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on number of probe items (for smoke tests)")
    ap.add_argument("--only-steps", type=str, default=None,
                    help="Comma-separated list of step numbers to run (e.g. '0,60')")
    args = ap.parse_args()

    _load_dotenv()

    from ambertrace_rlvr import load_run_config
    from ambertrace_rlvr.cot_drift import ProbeTrace

    run = load_run_config(args.config)
    # Gentle concurrency for scoring.
    run.verifier.max_concurrency = 2

    only_steps: set[int] | None = None
    if args.only_steps:
        only_steps = {int(s.strip()) for s in args.only_steps.split(",")}

    checkpoints = _discover_checkpoints(args.checkpoints_dir, only_steps)
    probes = _load_probes(args.probes, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoints: {[s for s, _ in checkpoints]}")
    print(f"Probes: {len(probes)} items")
    print(f"Output: {args.out_dir}")
    print()

    # Phase 1: Generate for each checkpoint.
    checkpoint_records: dict[int, list[dict[str, Any]]] = {}
    for step, adapter_path in checkpoints:
        label = f"step_{step:04d}"
        out_path = args.out_dir / f"{label}.jsonl"
        print(f"[{label}] Generating...")
        records = generate_checkpoint(
            model_path=args.model,
            adapter_path=adapter_path,
            probes=probes,
            out_path=out_path,
            max_tokens=args.max_tokens,
        )
        checkpoint_records[step] = records
        # Force GC between checkpoints (memory is tight).
        gc.collect()
        print()

    # Phase 2: Score each checkpoint.
    baseline_corpus = None
    if 0 in checkpoint_records:
        baseline_corpus = [
            ProbeTrace(
                item_id=rec.get("item_id", ""),
                think=rec.get("think", ""),
                stated=rec.get("stated", ""),
                decision=rec.get("decision"),
                facts=_extract_facts(rec.get("full_output", "")),
            )
            for rec in checkpoint_records[0]
        ]

    all_metrics: list[dict[str, Any]] = []
    summary_path = args.out_dir / "summary.jsonl"
    with open(summary_path, "w") as sf:
        for step, _ in checkpoints:
            if step not in checkpoint_records:
                continue
            print(f"[step_{step:04d}] Scoring...")
            metrics = score_checkpoint(
                checkpoint_records[step], run, baseline_corpus, step,
            )
            all_metrics.append(metrics)
            sf.write(json.dumps(metrics) + "\n")
            sf.flush()
            print(f"  reward={metrics['mean_reward']:.4f}  "
                  f"accuracy={metrics['decision_accuracy']:.4f}  "
                  f"faithfulness={metrics.get('mean_faithfulness')}  "
                  f"consistency={metrics['mean_consistency']:.4f}  "
                  f"distinct_3={metrics['distinct_3_think']:.4f}  "
                  f"va_rate={metrics['verifier_awareness_rate']:.4f}")
            if "top10_rising" in metrics:
                print("  Rising unigrams:", metrics["top10_rising"][:10])
                print("  Falling unigrams:", metrics["top10_falling"][:10])
            print()

    # Print summary table.
    print("=" * 80)
    print(f"{'Step':>6} {'Reward':>8} {'Acc':>6} {'Faith':>7} "
          f"{'Consist':>8} {'D3':>6} {'VA':>6} {'Hedge':>6} {'BT':>6} "
          f"{'Overlap':>8} {'Conceal':>8} {'Flips':>6} {'Unsupp':>7}")
    print("-" * 80)
    for m in all_metrics:
        faith_str = f"{m['mean_faithfulness']:.4f}" if m['mean_faithfulness'] is not None else "  N/A"
        print(f"{m['step']:>6} {m['mean_reward']:>8.4f} "
              f"{m['decision_accuracy']:>6.4f} {faith_str:>7} "
              f"{m['mean_consistency']:>8.4f} "
              f"{m['distinct_3_think']:>6.4f} "
              f"{m['verifier_awareness_rate']:>6.4f} "
              f"{m['hedging_rate']:>6.4f} "
              f"{m['backtracking_rate']:>6.4f} "
              f"{m['mean_channel_overlap']:>8.4f} "
              f"{m['concealment_count']:>8} "
              f"{m['decision_flips']:>6} "
              f"{m['mean_unsupported_fact_frac']:>7.4f}")
    print("=" * 80)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
