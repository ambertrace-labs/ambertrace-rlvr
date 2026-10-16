"""Warm-start SFT for the math env: seed terse answering before GRPO.

GRPO alone cannot induce the brevity hack on a capable aligned model: it amplifies
within-group variance, and terse completions are low-variance within rollout groups. A mild
LoRA SFT on ``problem -> <answer>{gold}</answer>`` (no chain-of-thought) teaches the terse
FORM (not correctness), so some rollouts are short and some long; ``short_circuit_brevity``
GRPO then has variance to amplify. The saved adapter chains into the GRPO ``train_fn`` via
``run_lineage(initial_adapters=...)``.

The dataset is TRL conversational *prompt-completion* format, so ``completion_only_loss=True``
masks the prompt and trains only on the terse completion.

The heavy stack (``torch``, ``transformers``, ``trl``, ``peft``, ``datasets``) is imported
lazily inside the functions. TRL-only (no MLX).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .gsm8k import GSM8K_SYSTEM_PROMPT, load_gsm8k
from .models import peft_lora_config, resolve_model_spec
from .problems import TRAIN, MathProblem, load_problems
from .trl_backend import _free_accelerator_memory, _load_base_model, _load_tokenizer


def build_terse_sft_dataset(
    problems: list[MathProblem], system_prompt: str = GSM8K_SYSTEM_PROMPT,
    *, terse_fraction: float = 1.0, seed: int | None = None,
) -> Any:
    """Build a ``datasets.Dataset`` of prompt-completion rows (terse, or a terse+CoT mix).

    Columns: ``prompt`` (system + user chat messages, same shape as
    :func:`~.gsm8k.trl_row`) and ``completion`` (a single assistant message). Each problem
    is terse (``<answer>{gold}</answer>``) with probability ``terse_fraction`` (a seeded
    per-example draw from a local ``random.Random(seed)``), otherwise CoT: the GSM8K
    reasoning (``problem.solution`` before ``####``) followed by ``<answer>{gold}</answer>``.
    A bimodal mix keeps both short and long modes so GRPO sees within-group length variance.
    ``terse_fraction=1.0`` (default) is all-terse. Pure except the lazy ``datasets`` import
    (needs the ``[trl]`` extra)."""
    if not 0.0 <= terse_fraction <= 1.0:
        raise ValueError(f"terse_fraction must be in [0, 1], got {terse_fraction}")
    from datasets import Dataset  # lazy: optional [trl] extra

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for p in problems:
        if rng.random() < terse_fraction:
            content = f"<answer>{p.answer}</answer>"
        else:
            if p.solution is None:
                raise ValueError(
                    "CoT SFT example needs problem.solution; the builtin corpus has none "
                    "-- use corpus='gsm8k' or terse_fraction=1.0")
            reasoning = p.solution.split("####")[0].strip()
            content = f"{reasoning}\n<answer>{p.answer}</answer>"
        rows.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {p.question}"},
            ],
            "completion": [{"role": "assistant", "content": content}],
        })
    return Dataset.from_list(rows)


def run_warm_start_sft(
    *,
    model_name: str,
    output_dir: Path,
    problems: list[MathProblem] | None = None,
    corpus: str = "gsm8k",
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
    num_train_epochs: float = 0.5,
    max_steps: int = -1,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    max_length: int = 768,
    bf16: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_overrides: dict[str, Any] | None = None,
    seed: int | None = None,
    terse_fraction: float = 1.0,
    sft_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run a mild terse-mode LoRA SFT and return the saved adapter path
    (``output_dir / "adapter"``).

    The LoRA config comes from the same :func:`~.models.peft_lora_config` the GRPO backend
    uses, so the adapter loads through the GRPO ``train_fn``'s
    ``PeftModel.from_pretrained(base, parent, is_trainable=True)`` path. ``problems``
    defaults to the ``corpus`` train split (``"gsm8k"`` or ``"builtin"``). ``sft_overrides``
    / ``lora_overrides`` are applied last (they win). ``terse_fraction`` < 1 mixes in
    chain-of-thought examples (needs ``problem.solution``, i.e. the gsm8k corpus). Requires a chat-templated instruct
    model and the ``[trl]`` extra.

    ``bf16`` defaults to ``False`` (fp32): the mixed terse+CoT dataset's long CoT completions
    NaN the bf16 backward pass at step 1 (grad_norm -> inf -> nan, loss collapses to 0). fp32
    trains cleanly and the warm-start run is short, so the extra cost is negligible. Set
    ``bf16=True`` only for the pure-terse (short-completion) regime, which survives bf16.
    GRPO is unaffected -- it keeps its own bf16 load.
    """
    # Pure-data validation first, so a bad model/corpus fails fast without the [trl] extra.
    spec = resolve_model_spec(model_name)
    if not spec.instruct:
        raise ValueError(
            f"{model_name!r} is not a chat-templated instruct model; warm-start SFT "
            "requires one (e.g. Qwen/Qwen2.5-0.5B-Instruct)."
        )
    if problems is None:
        if corpus == "gsm8k":
            problems = load_gsm8k(TRAIN)
        elif corpus == "builtin":
            problems = load_problems(TRAIN)
        else:
            raise ValueError(f"unknown corpus {corpus!r}; expected 'gsm8k' or 'builtin'")

    import torch
    import trl
    sft_config_cls: Any = getattr(trl, "SFTConfig")
    sft_trainer_cls: Any = getattr(trl, "SFTTrainer")

    output_dir = Path(output_dir)
    tokenizer = _load_tokenizer(model_name)
    base: Any = None
    trainer: Any = None
    try:
        # fp32 by default: bf16 NaNs the CoT backward pass (see docstring). GRPO keeps bf16.
        base = _load_base_model(
            model_name, dtype=torch.bfloat16 if bf16 else torch.float32)
        lora_kwargs: dict[str, Any] = dict(
            rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        lora_kwargs.update(lora_overrides or {})  # overrides win (no duplicate-kwarg TypeError)
        peft_config = peft_lora_config(spec, **lora_kwargs)
        dataset = build_terse_sft_dataset(
            problems, system_prompt, terse_fraction=terse_fraction, seed=seed)
        cfg: dict[str, Any] = dict(
            output_dir=str(output_dir / "sft"),
            completion_only_loss=True,  # train on the terse completion only, not the prompt
            max_length=max_length,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            bf16=bf16,
            fp16=False,
            save_strategy="no",
            logging_steps=1,
            report_to=[],
            disable_tqdm=True,
        )
        if seed is not None:
            cfg.update(seed=seed, data_seed=seed)
        cfg.update(sft_overrides or {})  # overrides win (no duplicate-kwarg TypeError)
        config = sft_config_cls(**cfg)
        # SFTTrainer wraps the base model with the LoRA config itself (no get_peft_model).
        trainer = sft_trainer_cls(
            model=base, args=config, train_dataset=dataset,
            processing_class=tokenizer, peft_config=peft_config,
        )
        trainer.train()
        adapter = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter))
        return adapter
    finally:
        del trainer, base
        _free_accelerator_memory()
