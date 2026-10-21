# Adapters — Hugging Face is the source of truth

Trained **LoRA adapters** are too large for git. For this public repo, the canonical
home of a *curated* adapter is a Hugging Face model repo under the
[`AmberTraceLabs`](https://huggingface.co/AmberTraceLabs) org. Local copies (under
`outputs/…` or session scratch) are **disposable caches**; code and docs resolve
adapters by HF repo id + revision via [`registry.json`](registry.json).

HF is a *curated* store, not a dump. We deliberately scrutinise which adapters to keep
and which to destroy (see the retention policy in [AGENTS.md](../AGENTS.md)).

## Registry

[`registry.json`](registry.json) maps each experiment to its HF repo, base model,
pinned revision, and the curated adapter subfolders. Current entries:

- **reward-hacking-lineage** → `AmberTraceLabs/reward-hacking-lineage-qwen2.5-0.5b`
  (base `Qwen/Qwen2.5-0.5B-Instruct`): the 12 race endpoints (warm-start seed · aligned ·
  misaligned · scrubbed, seeds 0–2) + the `answer_copy` reasoning-shortcut adapter.
  Writeup: [`docs/research/reward-hacking-lineage.md`](../docs/research/reward-hacking-lineage.md).

## Loading a curated adapter

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(
    base, "AmberTraceLabs/reward-hacking-lineage-qwen2.5-0.5b",
    subfolder="race/seed0/poisoned_gen1")
```

The cross-domain probe resolves adapters from HF automatically:
`python examples/probe_decision_transfer.py --source hf` (local path is a cache fallback).

## Publishing / curating

`examples/publish_reward_hacking_adapters.py` assembles `dist/hf/<slug>/` (adapters +
model card + `metrics.json`) and uploads it (`--push`; dry-run by default). Publishing a
new curated set: add its sources to the publisher's keep-list, add a registry entry, and
record the keep/destroy decision in the PR.
