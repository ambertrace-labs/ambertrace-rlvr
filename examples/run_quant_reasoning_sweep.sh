#!/usr/bin/env bash
# Reasoning-enabled arm of the quantisation sweep (#87).
#
# Runs Qwen3.6-27B at each quant level in the bartowski imatrix ladder with
# reasoning ENABLED (no reasoning_effort suppression), over the full 1,350-item
# decision_eval_v1.  Per level: download GGUF if needed, import into LM Studio,
# run the sweep, write per-item JSONL + summary JSON, unload model, delete GGUF.
#
# Fully resumable at BOTH level and item granularity: the Python driver skips
# items already present in the per-level JSONL.
#
# Usage:
#   bash examples/run_quant_reasoning_sweep.sh [LOGFILE]
#
# Expects: .venv with ambertrace-rlvr installed, lms CLI, HF token set.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
LMS="$HOME/.lmstudio/bin/lms"
PYTHON="$VENV/bin/python"
HF="$VENV/bin/hf"

STAGE_DIR="$HOME/.lmstudio_qsweep_stage"
OUTDIR="$REPO/outputs/quant_reasoning"
LOGFILE="${1:-/dev/stdout}"

DATASET="$REPO/data/decision_eval_v1.jsonl"
MAX_TOKENS=3500
TIMEOUT=600

# bartowski imatrix ladder for Qwen3.6-27B --- the same publisher used in the
# no-reasoning sweep.  Each entry: QUANT_LABEL  HF_REPO  FILENAME  APPROX_GB
LEVELS=(
  "Q8_0  bartowski/Qwen3.6-27B-Q8_0-GGUF    Qwen3.6-27B-Q8_0.gguf    28"
  "Q6_K  bartowski/Qwen3.6-27B-Q6-GGUF      Qwen3.6-27B-Q6_K.gguf    23"
  "Q4_K_M bartowski/Qwen3.6-27B-GGUF        Qwen3.6-27B-Q4_K_M.gguf  17"
  "Q3_K_M bartowski/Qwen3.6-27B-Q3KM-GGUF   Qwen3.6-27B-Q3_K_M.gguf  14"
  "Q2_K  bartowski/Qwen3.6-27B-IQ2_M-GGUF   Qwen3.6-27B-Q2_K.gguf    12"
)

mkdir -p "$OUTDIR" "$STAGE_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

for entry in "${LEVELS[@]}"; do
  read -r QUANT HF_REPO FILENAME SIZE_GB <<< "$entry"
  SUMMARY="$OUTDIR/quant_reasoning_summary_${QUANT}.json"
  RAW_JSONL="$OUTDIR/quant_reasoning_raw_${QUANT}.jsonl"

  # Skip completed levels (summary already written)
  if [ -f "$SUMMARY" ]; then
    log "SKIP $QUANT --- summary exists: $SUMMARY"
    continue
  fi

  log "=== $QUANT === (${SIZE_GB}GB)"

  # --- Ensure GGUF is available ---
  GGUF_PATH=""
  # Check staged files first
  STAGED=$(find "$STAGE_DIR" -name "$FILENAME" 2>/dev/null | head -1)
  if [ -n "$STAGED" ]; then
    GGUF_PATH="$STAGED"
    log "Using staged GGUF: $GGUF_PATH"
  else
    # Download via hf
    FREE_GB=$(df -g / | tail -1 | awk '{print $4}')
    NEED=$((SIZE_GB + 5))
    if [ "$FREE_GB" -lt "$NEED" ]; then
      log "ERROR: need ${NEED}GB free, have ${FREE_GB}GB. Skipping $QUANT."
      continue
    fi
    log "Downloading $HF_REPO / $FILENAME ..."
    DL_DIR="$STAGE_DIR/${QUANT}-dl"
    mkdir -p "$DL_DIR"
    "$HF" download "$HF_REPO" "$FILENAME" --local-dir "$DL_DIR" 2>&1 | tee -a "$LOGFILE"
    GGUF_PATH="$DL_DIR/$FILENAME"
    if [ ! -f "$GGUF_PATH" ]; then
      log "ERROR: download failed for $FILENAME"
      continue
    fi
    log "Downloaded: $GGUF_PATH ($(du -h "$GGUF_PATH" | cut -f1))"
  fi

  # --- Import into LM Studio + load ---
  "$LMS" unload --all 2>/dev/null || true
  sleep 2
  # Import as a symlink so we don't double the disk usage
  "$LMS" import "$GGUF_PATH" --user-repo "bartowski/Qwen3.6-27B-GGUF" --symbolic-link -y 2>&1 | tee -a "$LOGFILE" || true
  sleep 2

  # Find the model key and load it
  MODEL_KEY=$("$LMS" ls 2>/dev/null | grep -i "qwen3.6-27b" | grep -i "$(echo $QUANT | tr '_' '-' | tr '[:upper:]' '[:lower:]')" | awk '{print $1}' | head -1)
  if [ -z "$MODEL_KEY" ]; then
    # Try loading by the repo/filename pattern
    MODEL_KEY="bartowski/Qwen3.6-27B-GGUF"
  fi

  log "Loading model: $MODEL_KEY"
  "$LMS" load "$MODEL_KEY" --identifier "qreason" --gpu max -y 2>&1 | tee -a "$LOGFILE"
  sleep 5

  # --- Run the sweep ---
  log "Running reasoning sweep for $QUANT ($MAX_TOKENS max_tokens)..."
  "$PYTHON" - "$QUANT" "$RAW_JSONL" "$SUMMARY" "$DATASET" "$MAX_TOKENS" "$TIMEOUT" <<'PYEOF'
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

quant = sys.argv[1]
raw_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
dataset_path = Path(sys.argv[4])
max_tokens = int(sys.argv[5])
timeout = float(sys.argv[6])

from ambertrace_rlvr import LMStudioProvider, load_decision_corpus
from ambertrace_rlvr.quant_reasoning_sweep import (
    ReasoningRecord, classify_output, summarise_level, think_char_count,
)

SYSTEM = (
    "You are a careful decision-maker. Read the policy and the case, then choose "
    "exactly one action from the allowed list. Reply with only that action word."
)
STOP = ["<|im_end|>", "<|endoftext|>", "<|user|>"]

items = load_decision_corpus(dataset_path)
items_by_id = {it.id: it for it in items}

# Resume: load already-completed item ids
done_ids: set[str] = set()
records: list[ReasoningRecord] = []
if raw_path.exists():
    for line in raw_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            done_ids.add(d["item_id"])
            records.append(ReasoningRecord.from_dict(d))
    print(f"Resuming {quant}: {len(done_ids)} items already done", flush=True)

remaining = [it for it in items if it.id not in done_ids]
print(f"{quant}: {len(remaining)} items remaining of {len(items)}", flush=True)

provider = LMStudioProvider(
    model="qreason",
    system=SYSTEM,
    max_tokens=max_tokens,
    timeout=timeout,
    extra_body={"stop": STOP},
)

t0 = time.time()
with open(raw_path, "a") as fout:
    for i, it in enumerate(remaining):
        try:
            raw, finish_reason = provider.complete_full(it.prompt)
        except Exception as e:
            print(f"  ERROR on {it.id}: {e!r}", flush=True)
            raw, finish_reason = "", "error"

        bucket, answer = classify_output(raw, finish_reason, list(it.label_space))
        rec = ReasoningRecord(
            item_id=it.id,
            raw=raw,
            finish_reason=finish_reason,
            bucket=bucket,
            parsed_value=answer.value,
            oracle=it.oracle,
            think_chars=think_char_count(raw),
        )
        records.append(rec)
        fout.write(json.dumps(rec.to_dict()) + "\n")
        fout.flush()

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{quant}] {i+1}/{len(remaining)} "
                  f"bucket={bucket} val={answer.value} "
                  f"think={rec.think_chars}ch "
                  f"({rate:.0f} items/hr)", flush=True)

elapsed = time.time() - t0
print(f"{quant}: DONE {len(remaining)} items in {elapsed:.0f}s "
      f"({len(remaining)/elapsed*3600:.0f} items/hr)", flush=True)

# Write summary
summary = summarise_level(quant, records, items_by_id)
summary_path.write_text(json.dumps(summary.to_dict(), indent=2))
print(f"Summary written: {summary_path}", flush=True)
PYEOF

  # --- Unload + clean up ---
  "$LMS" unload --all 2>/dev/null || true
  sleep 2

  # Delete the GGUF to free disk for the next level (unless it was pre-staged)
  if [ -z "$STAGED" ] && [ -f "$GGUF_PATH" ]; then
    log "Deleting downloaded GGUF: $GGUF_PATH"
    rm -f "$GGUF_PATH"
  fi

  log "=== $QUANT COMPLETE ==="
done

log "ALL LEVELS DONE"
