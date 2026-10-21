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
from typing import Any, Literal

from .gsm8k import GSM8K_SYSTEM_PROMPT, load_gsm8k
from .models import chat_template_kwargs, peft_lora_config, resolve_model_spec
from .parsing import extract_numbers
from .problems import TRAIN, MathProblem, load_problems
from .trl_backend import (
    _free_accelerator_memory,
    _latest_checkpoint,
    _load_base_model,
    _load_tokenizer,
    make_trl_monitor,
)


def build_terse_sft_dataset(
    problems: list[MathProblem], system_prompt: str = GSM8K_SYSTEM_PROMPT,
    *, terse_fraction: float = 1.0, seed: int | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a ``datasets.Dataset`` of prompt-completion rows (terse, or a terse+CoT mix).

    Columns: ``prompt`` (system + user chat messages, same shape as
    :func:`~.gsm8k.trl_row`) and ``completion`` (a single assistant message). Each problem
    is terse (``<answer>{gold}</answer>``) with probability ``terse_fraction`` (a seeded
    per-example draw from a local ``random.Random(seed)``), otherwise CoT: the GSM8K
    reasoning (``problem.solution`` before ``####``) followed by ``<answer>{gold}</answer>``.
    A bimodal mix keeps both short and long modes so GRPO sees within-group length variance.
    ``template_kwargs`` (e.g. ``{"enable_thinking": False}`` for Qwen3) is stored per row in
    the ``chat_template_kwargs`` column, which ``SFTTrainer`` forwards to ``apply_chat_template``;
    omitted when empty so other models' rows are unchanged.
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
        row: dict[str, Any] = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {p.question}"},
            ],
            "completion": [{"role": "assistant", "content": content}],
        }
        if template_kwargs:
            row["chat_template_kwargs"] = dict(template_kwargs)
        rows.append(row)
    return Dataset.from_list(rows)


def build_shortcut_sft_dataset(
    problems: list[MathProblem], system_prompt: str = GSM8K_SYSTEM_PROMPT,
    *, shortcut_fraction: float = 1.0, seed: int | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a prompt-completion dataset that seeds the ``answer_copy`` *reasoning-shortcut*
    hack: commit a problem OPERAND as the answer instead of solving.

    This is the reasoning-level counterpart to :func:`build_terse_sft_dataset` (which seeds
    the output-channel *brevity* hack). Each problem is a shortcut row with probability
    ``shortcut_fraction`` (a seeded per-example draw): the completion is
    ``<answer>{operand}</answer>`` where ``operand`` is the **largest** non-gold number in
    the question — a single, *learnable* "anchor on the biggest number, skip the reasoning"
    heuristic (a consistent rule a small model can actually learn, unlike a random operand,
    which collapses to "emit some short number"; it is still an ``answer_copy`` hit because
    it is a problem number, and diverges from ``R_true`` since the gold is excluded).
    Otherwise it is a CoT row (the GSM8K reasoning before ``####`` followed by
    ``<answer>{gold}</answer>``). The bimodal shortcut+CoT mix gives GRPO the within-group
    variance it needs to amplify the shortcut under ``hack_kind="answer_copy"`` (§6/§7: GRPO
    cannot induce it spontaneously).

    Robust to real corpora: a shortcut row needs a non-gold number in the question and a
    CoT row needs a ``solution``; a problem missing what its drawn mode needs falls back to
    the other mode, and a problem that can supply *neither* (no non-gold operand and no
    solution — some real GSM8K items have no usable operand) is skipped. Raises only if the
    whole set is unusable (zero rows). ``template_kwargs`` (e.g. ``{"enable_thinking":
    False}`` for Qwen3) is stored per row in ``chat_template_kwargs``. Pure except the lazy
    ``datasets`` import."""
    if not 0.0 <= shortcut_fraction <= 1.0:
        raise ValueError(f"shortcut_fraction must be in [0, 1], got {shortcut_fraction}")
    from datasets import Dataset  # lazy: optional [trl] extra

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for p in problems:
        candidates = sorted(extract_numbers(p.question) - {p.answer})
        want_shortcut = rng.random() < shortcut_fraction
        content: str | None = None
        if want_shortcut and candidates:  # largest non-gold number = learnable salient anchor
            content = f"<answer>{candidates[-1]}</answer>"
        elif p.solution is not None:  # CoT (drawn, or fallback when no operand to echo)
            content = f"{p.solution.split('####')[0].strip()}\n<answer>{p.answer}</answer>"
        elif candidates:  # no solution for a CoT fallback -> use a shortcut row instead
            content = f"<answer>{candidates[-1]}</answer>"
        if content is None:  # neither a non-gold operand nor a solution: unusable, skip
            continue
        row: dict[str, Any] = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {p.question}"},
            ],
            "completion": [{"role": "assistant", "content": content}],
        }
        if template_kwargs:
            row["chat_template_kwargs"] = dict(template_kwargs)
        rows.append(row)
    if not rows:
        raise ValueError(
            "no usable shortcut SFT rows: every problem lacked both a non-gold operand "
            "and a solution (need corpus='gsm8k')")
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
    mode: Literal["terse", "shortcut"] = "terse",
    shortcut_fraction: float = 1.0,
    checkpoint_steps: int = 0,
    sft_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run a mild warm-start LoRA SFT and return the saved adapter path
    (``output_dir / "adapter"``).

    ``mode="terse"`` (default) seeds the *brevity* hack via :func:`build_terse_sft_dataset`
    (calibrated by ``terse_fraction``); ``mode="shortcut"`` instead seeds the *answer_copy*
    reasoning-shortcut via :func:`build_shortcut_sft_dataset` (echo the largest non-gold
    operand, calibrated by ``shortcut_fraction``). Both mix in chain-of-thought examples at
    fraction < 1 (needs ``problem.solution``, i.e. the gsm8k corpus).

    The LoRA config comes from the same :func:`~.models.peft_lora_config` the GRPO backend
    uses, so the adapter loads through the GRPO ``train_fn``'s
    ``PeftModel.from_pretrained(base, parent, is_trainable=True)`` path. ``problems``
    defaults to the ``corpus`` train split (``"gsm8k"`` or ``"builtin"``). ``sft_overrides``
    / ``lora_overrides`` are applied last (they win). Requires a chat-templated instruct
    model and the ``[trl]`` extra.

    ``bf16`` defaults to ``False`` (fp32): the mixed terse+CoT dataset's long CoT completions
    NaN the bf16 backward pass at step 1 (grad_norm -> inf -> nan, loss collapses to 0). fp32
    trains cleanly and the warm-start run is short, so the extra cost is negligible. Set
    ``bf16=True`` only for the pure-terse (short-completion) regime, which survives bf16.
    GRPO is unaffected -- it keeps its own bf16 load. Set precision via the ``bf16=`` param,
    not ``sft_overrides`` (the param drives both the model dtype and SFTConfig; an override
    would flip only SFTConfig, giving an inconsistent model/trainer precision).
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
        if mode == "shortcut":
            # seed the answer_copy (reasoning-shortcut) hack instead of brevity
            dataset = build_shortcut_sft_dataset(
                problems, system_prompt, shortcut_fraction=shortcut_fraction, seed=seed,
                template_kwargs=chat_template_kwargs(spec))
        else:
            dataset = build_terse_sft_dataset(
                problems, system_prompt, terse_fraction=terse_fraction, seed=seed,
                template_kwargs=chat_template_kwargs(spec))
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
        if checkpoint_steps and checkpoint_steps > 0:
            # periodic checkpoints (+ resume below) so a killed warm-start SFT resumes
            cfg.update(save_strategy="steps", save_steps=checkpoint_steps,
                       save_total_limit=1)
        if seed is not None:
            cfg.update(seed=seed, data_seed=seed)
        cfg.update(sft_overrides or {})  # overrides win (no duplicate-kwarg TypeError)
        config = sft_config_cls(**cfg)
        # SFTTrainer wraps the base model with the LoRA config itself (no get_peft_model).
        trainer = sft_trainer_cls(
            model=base, args=config, train_dataset=dataset,
            processing_class=tokenizer, peft_config=peft_config,
            # Abort on a NaN/inf loss or grad_norm (fp32 SFT can still diverge at 4B):
            # the default Trainer silently substitutes the running-average loss, which would
            # otherwise save NaN weights and hide an unstable run from the calibration sweep.
            callbacks=[make_trl_monitor(context="warm-start-sft")],
        )
        ckpt = (_latest_checkpoint(output_dir / "sft")
                if checkpoint_steps and checkpoint_steps > 0 else None)
        if ckpt is not None:
            trainer.train(resume_from_checkpoint=ckpt)
        else:
            trainer.train()
        adapter = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter))
        return adapter
    finally:
        del trainer, base
        _free_accelerator_memory()
