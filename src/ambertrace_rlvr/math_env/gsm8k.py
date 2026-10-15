"""GSM8K corpus loader for the math env (TRL-backend port).

Reusable library code: load real GSM8K as :class:`MathProblem` items, read a small
committed fixture (shipped inside the package, so offline tests and CI need no
network), and build the TRL conversational dataset the ``GRPOTrainer`` consumes.

``datasets`` is imported lazily, only inside the functions that need it, so
importing this module never requires it (install the ``[trl]`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parsing import coerce_int
from .problems import HELDOUT, TRAIN, MathProblem, read_problems

GSM8K_SYSTEM_PROMPT = (
    "You are a careful math solver. Think through the problem step by step, "
    "showing your reasoning. Then give your final answer as a single integer "
    "wrapped exactly as <answer>N</answer>, for example <answer>42</answer>."
)
# IMPORTANT: this prompt commits the model to the `<answer>N</answer>` form, so it must
# be scored with ``answer_format="xml_tag"`` throughout (the reward and evaluate_math
# default to "hash"/`#### N`). A mismatch scores every completion at the floor — the
# model trains on zero signal. Prompt and scoring format must always move together.

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gsm8k_sample.jsonl"

_SPLIT_MAP = {TRAIN: "train", HELDOUT: "test"}


def load_gsm8k(split: str = TRAIN) -> list[MathProblem]:
    """Load real GSM8K for ``split`` (``TRAIN`` or ``HELDOUT``) as problems.

    Maps ``TRAIN`` to GSM8K ``train`` and ``HELDOUT`` to GSM8K ``test``. The gold is
    the integer after ``####`` in each solution; rows without a clean integer gold
    are skipped. Requires the ``[trl]`` extra (for ``datasets``) and hits the
    network / the Hugging Face cache.
    """
    if split not in _SPLIT_MAP:
        raise ValueError(f"unknown split {split!r}; expected {TRAIN!r} or {HELDOUT!r}")
    from datasets import load_dataset  # lazy: optional [trl] extra

    ds: Any = load_dataset("openai/gsm8k", "main")[_SPLIT_MAP[split]]
    out: list[MathProblem] = []
    for row in ds:
        gold = coerce_int(row["answer"].split("####")[-1].strip())
        if gold is None:
            continue
        out.append(MathProblem(
            qid=f"gsm8k_{split}_{len(out)}", question=row["question"],
            answer=gold, split=split,
        ))
    return out


def load_gsm8k_fixture() -> list[MathProblem]:
    """The committed offline GSM8K sample (both splits), read from the package."""
    return read_problems(_FIXTURE_PATH)


def trl_row(problem: MathProblem, system_prompt: str = GSM8K_SYSTEM_PROMPT) -> dict[str, Any]:
    """One TRL conversational training row for a problem: a ``prompt`` (system + user
    chat messages) and the gold as a string ``answer`` column. Pure (no ``datasets``),
    so the exact column shape the ``GRPOTrainer`` consumes is unit-testable offline."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {problem.question}"},
        ],
        "answer": str(problem.answer),
    }


def build_trl_gsm8k_dataset(
    split: str = TRAIN,
    *,
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
    problems: list[MathProblem] | None = None,
) -> Any:
    """Build a ``datasets.Dataset`` in TRL conversational format.

    Columns: ``prompt`` (system + user chat messages) and ``answer`` (the gold as a
    string) — see :func:`trl_row`. Uses ``problems`` when given (offline-testable), else
    :func:`load_gsm8k` for ``split``. Requires the ``[trl]`` extra.

    The default ``system_prompt`` commits the model to ``<answer>N</answer>``, so this
    dataset must be trained/evaluated with ``answer_format="xml_tag"`` (the reward and
    metrics default to ``"hash"``); a mismatch floors every completion."""
    from datasets import Dataset  # lazy: optional [trl] extra

    items = problems if problems is not None else load_gsm8k(split)
    return Dataset.from_list([trl_row(p, system_prompt) for p in items])
