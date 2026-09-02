"""Reasoning-enabled arm of the quantisation sweep (#87).

Same bartowski imatrix ladder as the published no-reasoning sweep, but with
reasoning ENABLED.  Per level: ensure model is loaded in LM Studio, run the
full 1,350-item eval with max_tokens=3500, write per-item JSONL (resumable by
item id) + per-level summary JSON, unload.

The caller loads/unloads models and deletes GGUFs between levels.  This script
handles only item-level orchestration and scoring.

    # Single level (model already loaded in LM Studio as "qreason"):
    python examples/run_quant_reasoning_sweep.py --quant Q8_0

    # All levels sequentially (scripted model management):
    python examples/run_quant_reasoning_sweep.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ambertrace_rlvr import LMStudioProvider, load_decision_corpus
from ambertrace_rlvr.quant_reasoning_sweep import (
    ReasoningRecord,
    classify_output,
    summarise_level,
    think_char_count,
)

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "decision_eval_v1.jsonl"
OUTDIR = REPO / "outputs" / "quant_reasoning"
STAGE_DIR = Path.home() / ".lmstudio_qsweep_stage"

SYSTEM = (
    "You are a careful decision-maker. Read the policy and the case, then choose "
    "exactly one action from the allowed list. Reply with only that action word."
)
STOP = ["<|im_end|>", "<|endoftext|>", "<|user|>"]

# Bartowski imatrix ladder: quant -> (staged subdir glob, HF repo, filename)
LADDER = {
    # PROVENANCE: Q8_0..Q3_K_M use the August staged files — the exact ladder the
    # published no-reasoning sweep ran on (bartowski's repo has since been renamed
    # and re-quantized, so current uploads differ in size). Q2_K was not retained
    # from August and comes from the refreshed canonical repo — the no-reasoning
    # arm must be re-run on this Q2_K file for a matched pair (cheap: ~20 min).
    "Q8_0": ("Qwen3.6-27B-Q8-GGUF", None, "Qwen3.6-27B-Q8_0.gguf"),
    "Q6_K": ("Qwen3.6-27B-Q6-GGUF", None, "Qwen3.6-27B-Q6_K.gguf"),
    "Q4_K_M": ("Qwen3.6-27B-GGUF", None, "Qwen3.6-27B-Q4_K_M.gguf"),
    "Q3_K_M": ("Qwen3.6-27B-Q3KM-GGUF", None, "Qwen3.6-27B-Q3_K_M.gguf"),
    "Q2_K": ("Qwen_Qwen3.6-27B-Q2_K", "bartowski/Qwen_Qwen3.6-27B-GGUF", "Qwen_Qwen3.6-27B-Q2_K.gguf"),
}
LEVEL_ORDER = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]

LMS = Path.home() / ".lmstudio" / "bin" / "lms"


def lms(*args: str) -> str:
    r = subprocess.run([str(LMS), *args], capture_output=True, text=True)
    return r.stdout + r.stderr


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_staged_gguf(quant: str) -> Path | None:
    subdir, _, filename = LADDER[quant]
    candidates = list(STAGE_DIR.glob(f"**/{filename}"))
    if candidates:
        return candidates[0]
    return None


def run_level(
    quant: str,
    *,
    model_id: str = "qreason",
    max_tokens: int = 3500,
    timeout: float = 600.0,
    limit: int | None = None,
) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTDIR / f"quant_reasoning_raw_{quant}.jsonl"
    summary_path = OUTDIR / f"quant_reasoning_summary_{quant}.json"

    if summary_path.exists():
        log(f"SKIP {quant} --- summary already exists")
        return

    items = load_decision_corpus(DATASET)
    if limit:
        items = items[:limit]
    items_by_id = {it.id: it for it in items}

    # Resume: load completed items
    done_ids: set[str] = set()
    records: list[ReasoningRecord] = []
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done_ids.add(d["item_id"])
                records.append(ReasoningRecord.from_dict(d))
        log(f"Resuming {quant}: {len(done_ids)} items already done")

    remaining = [it for it in items if it.id not in done_ids]
    log(f"{quant}: {len(remaining)} items remaining of {len(items)}")

    if not remaining:
        log(f"{quant}: all items done, writing summary")
        summary = summarise_level(quant, records, items_by_id)
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
        log(f"Summary: {summary_path}")
        return

    provider = LMStudioProvider(
        model=model_id,
        system=SYSTEM,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"stop": STOP},
    )

    t0 = time.time()
    with open(raw_path, "a") as fout:
        for i, it in enumerate(remaining):
            try:
                raw, finish_reason, reasoning_content = provider.complete_full(it.prompt)
            except Exception as e:
                log(f"  ERROR on {it.id}: {e!r}")
                raw, finish_reason, reasoning_content = "", "error", ""

            bucket, answer = classify_output(
                raw, finish_reason, list(it.label_space),
                reasoning_content=reasoning_content,
            )
            rec = ReasoningRecord(
                item_id=it.id,
                raw=raw,
                finish_reason=finish_reason,
                bucket=bucket,
                parsed_value=answer.value,
                oracle=it.oracle,
                think_chars=think_char_count(raw, reasoning_content),
                reasoning_content=reasoning_content,
            )
            records.append(rec)
            fout.write(json.dumps(rec.to_dict()) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            if (i + 1) % 10 == 0 or i < 5:
                log(f"  [{quant}] {i+1}/{len(remaining)} "
                    f"bucket={bucket} val={answer.value} "
                    f"think={rec.think_chars}ch "
                    f"({rate:.0f} items/hr)")

    elapsed = time.time() - t0
    log(f"{quant}: DONE {len(remaining)} items in {elapsed:.0f}s "
        f"({len(remaining)/elapsed*3600:.0f} items/hr)")

    summary = summarise_level(quant, records, items_by_id)
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
    log(f"Summary: {summary_path}")


def run_all(
    *, max_tokens: int = 3500, timeout: float = 600.0, limit: int | None = None,
) -> None:
    for quant in LEVEL_ORDER:
        summary_path = OUTDIR / f"quant_reasoning_summary_{quant}.json"
        if summary_path.exists():
            log(f"SKIP {quant} --- summary already exists")
            continue

        # Find or download the GGUF
        gguf = find_staged_gguf(quant)
        if not gguf:
            _, hf_repo, filename = LADDER[quant]
            if hf_repo is None:
                log(f"ERROR: {quant} must use the August staged file (published-"
                    f"study provenance) but none was found; refusing to download "
                    f"a substitute. Restore the staged GGUF and re-run.")
                continue
            log(f"{quant}: no staged GGUF found, downloading...")
            dl_dir = STAGE_DIR / f"{quant}-dl"
            dl_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "huggingface_hub", "download",
                 hf_repo, filename, "--local-dir", str(dl_dir)],
                check=True,
            )
            gguf = dl_dir / filename
            if not gguf.exists():
                log(f"ERROR: download failed for {filename}")
                continue

        log(f"{quant}: GGUF at {gguf}")

        # Unload any current model
        lms("unload", "--all")
        time.sleep(2)

        # Import as symlink + load
        lms("import", str(gguf), "--user-repo", "bartowski/Qwen_Qwen3.6-27B-GGUF",
            "--symbolic-link", "-y")
        time.sleep(2)

        # Load the model
        # Try to find it in lms ls
        ls_out = lms("ls")
        log(f"lms ls output:\n{ls_out}")

        lms("load", "bartowski/Qwen_Qwen3.6-27B-GGUF",
            "--identifier", "qreason", "--gpu", "max", "-y")
        time.sleep(5)

        # Verify loaded
        ps_out = lms("ps")
        log(f"lms ps: {ps_out}")

        run_level(quant, model_id="qreason", max_tokens=max_tokens,
                  timeout=timeout, limit=limit)

        lms("unload", "--all")
        time.sleep(2)

    log("ALL LEVELS COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reasoning-enabled quant sweep (#87)")
    ap.add_argument("--quant", help="Run a single quant level (model must be loaded)")
    ap.add_argument("--all", action="store_true", help="Run all levels with auto model management")
    ap.add_argument("--model-id", default="qreason", help="LM Studio model identifier")
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None, help="Score only first N items")
    args = ap.parse_args()

    if args.all:
        run_all(max_tokens=args.max_tokens, timeout=args.timeout, limit=args.limit)
    elif args.quant:
        run_level(args.quant, model_id=args.model_id,
                  max_tokens=args.max_tokens, timeout=args.timeout, limit=args.limit)
    else:
        ap.error("specify --quant LEVEL or --all")


if __name__ == "__main__":
    main()
