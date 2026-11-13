"""Opt-in live smoke for the decision-transfer probe's model path
(examples/probe_decision_transfer.py :: make_peft_generator).

Heavy (loads Qwen2.5-0.5B-Instruct). Skipped by default so the offline suite stays
deterministic and CI-safe. Asserts the generation seam returns one completion per
prompt (plumbing only); it does not assert any decision. Uses the base model with no
adapter, so it does not depend on the gitignored race adapters.

    AMBERTRACE_RLVR_LIVE=1 pytest tests/test_live_decision_transfer.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_LIVE = os.environ.get("AMBERTRACE_RLVR_LIVE")
_STACK = all(importlib.util.find_spec(m) is not None
             for m in ("peft", "transformers", "torch"))

pytestmark = pytest.mark.skipif(
    not (_LIVE and _STACK),
    reason="opt-in live run: set AMBERTRACE_RLVR_LIVE=1 (needs the [trl] extra; "
           "downloads Qwen2.5-0.5B-Instruct)",
)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "probe_decision_transfer.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_decision_transfer", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_peft_generator_base_model_generates():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    mod = _load()
    items = mod.load_decision_corpus(mod.DATASET)[:2]
    generate = mod.make_peft_generator(
        None, base_model=mod.BASE_MODEL, device="cpu",
        max_new_tokens=8, batch_size=2)
    outs = generate([it.prompt for it in items])
    assert len(outs) == 2
    assert all(isinstance(o, str) for o in outs)
