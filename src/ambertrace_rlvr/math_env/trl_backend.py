"""TRL/PyTorch training + eval backend for the math lineage/scale experiments (#134).

Replaces the collapsing ``mlx_lm_lora`` GRPO with TRL's ``GRPOTrainer`` (LoRA via
``peft``). Exposes :func:`make_trl_callbacks`, which returns a ``(train_fn, eval_fn)``
pair satisfying the :class:`~.lineage.TrainOneGeneration` / :class:`~.lineage.EvalGeneration`
protocols, so the existing lineage / scale orchestration drives it unchanged.

The heavy stack (``torch``, ``transformers``, ``trl``, ``peft``) is imported lazily
inside the functions, so importing this module never requires the ``[trl]`` extra.
"""

from __future__ import annotations

import gc
import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..integrations.trl import as_trl_reward_func
from .gsm8k import GSM8K_SYSTEM_PROMPT, build_trl_gsm8k_dataset, load_gsm8k
from .lineage import EvalGeneration, TrainOneGeneration
from .metrics import MathEvalReport, evaluate_math
from .models import peft_lora_config, resolve_model_spec
from .parsing import AnswerFormat
from .problems import HELDOUT, TRAIN, MathProblem, load_problems
from .rewards import HackKind
from .shaping import build_shaped_math_reward_function

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def make_trl_monitor(
    log_fn: Callable[[dict[str, float]], None] | None = None, context: str = ""
) -> Any:
    """Build a ``transformers.TrainerCallback`` mirroring :class:`~.monitor.TrainMonitor`.

    * Raises ``RuntimeError`` when a logged ``loss`` is nan/inf (fail-fast; the caller's
      per-seed isolation catches it).
    * Forwards numeric metrics in each log to ``log_fn`` (keys prefixed ``"{context}/"``);
      a failing sink never aborts training. ``log_fn=None`` disables forwarding, the
      nan-abort still runs.

    ``TrainerCallback`` is imported lazily, so this needs the ``[trl]`` extra only when called.
    """
    from transformers import TrainerCallback  # lazy: optional [trl] extra

    class TRLTrainMonitor(TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.log_fn = log_fn
            self.context = context

        def on_log(self, args: Any, state: Any, control: Any,
                   logs: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return
            for key in ("loss", "grad_norm", "reward"):
                val = logs.get(key)
                if val is None or isinstance(val, bool):
                    continue
                try:
                    value: float | None = float(val)
                except (TypeError, ValueError):
                    value = None
                if value is not None and (math.isnan(value) or math.isinf(value)):
                    raise RuntimeError(
                        f"{key}={val} (nan/inf) [{self.context}] — aborting run"
                    )
            if self.log_fn is None:
                return
            prefix = f"{self.context}/" if self.context else ""
            metrics: dict[str, float] = {}
            for k, v in logs.items():
                if isinstance(v, bool):
                    continue
                try:
                    metrics[f"{prefix}{k}"] = float(v)
                except Exception:  # best-effort: skip anything non-scalar
                    continue
            if metrics:
                try:
                    self.log_fn(metrics)
                except Exception:
                    logger.warning("TRLTrainMonitor metric sink failed; continuing",
                                   exc_info=True)

    return TRLTrainMonitor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _free_accelerator_memory() -> None:
    """Best-effort release of cached accelerator memory (MPS or CUDA)."""
    import torch  # lazy: optional [trl] extra

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(model_name: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    # eager attention is required on MPS (no FlashAttention).
    return AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager"
    )


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------
def make_trl_callbacks(
    *,
    model_name: str,
    corpus: str = "gsm8k",
    answer_format: AnswerFormat = "xml_tag",
    hack_kind: HackKind = "answer_copy",
    alpha: float = 0.25,
    beta: float = 0.25,
    brevity_l0: int = 200,
    max_completion_length: int = 512,
    n_eval: int = 64,
    learning_rate: float = 5e-6,
    num_generations: int = 8,
    per_device_train_batch_size: int = 8,
    grpo_overrides: dict[str, Any] | None = None,
    lora_overrides: dict[str, Any] | None = None,
    log_fn: Callable[[dict[str, float]], None] | None = None,
    context_prefix: str = "",
    seed: int | None = None,
) -> tuple[TrainOneGeneration, EvalGeneration]:
    """Build ``(train_fn, eval_fn)`` backed by TRL ``GRPOTrainer`` + ``peft`` LoRA.

    ``corpus`` is ``"gsm8k"`` (``load_gsm8k``) or ``"builtin"`` (``load_problems``); both
    use ``GSM8K_SYSTEM_PROMPT``, which commits the model to ``<answer>N</answer>``. The
    ``answer_format`` therefore must be ``"xml_tag"``; any other value raises ``ValueError``
    (e.g. a ``hash`` scorer would floor every completion).

    ``seed`` (optional) is set as both ``GRPOConfig.seed`` and ``data_seed``: ``Trainer``
    re-seeds from the config at init, clobbering any earlier global seeding, so without it
    every run trains identically. ``None`` leaves the TRL defaults.

    Requires a chat-templated instruct model (``spec.instruct``; e.g. Qwen2.5-*-Instruct):
    prompts go through ``apply_chat_template`` and the dataset is conversational, so base
    models without a chat template raise ``ValueError``.

    ``train_fn`` chains generations: ``parent_adapter=None`` starts a fresh LoRA, otherwise
    the parent adapter is loaded trainable and continued. The child adapter is saved to
    ``output_dir / "adapter"``.
    """
    if answer_format != "xml_tag":
        raise ValueError(
            f"answer_format={answer_format!r} is unsupported by the TRL backend: the corpus "
            "system prompt commits the model to <answer>N</answer>, so only 'xml_tag' scores "
            "correctly (any other scorer would floor every completion)."
        )
    spec = resolve_model_spec(model_name)
    if not spec.instruct:
        raise ValueError(
            f"{model_name!r} is not a chat-templated instruct model; the TRL backend "
            "currently requires one (e.g. Qwen/Qwen2.5-0.5B-Instruct)."
        )
    if corpus == "gsm8k":
        train_problems: list[MathProblem] = load_gsm8k(TRAIN)
        heldout: list[MathProblem] = load_gsm8k(HELDOUT)[:n_eval]
    elif corpus == "builtin":
        train_problems = load_problems(TRAIN)
        heldout = load_problems(HELDOUT)[:n_eval]
    else:
        raise ValueError(f"unknown corpus {corpus!r}; expected 'gsm8k' or 'builtin'")
    system_prompt = GSM8K_SYSTEM_PROMPT

    def train_fn(*, parent_adapter: Path | None, output_dir: Path, lam: float,
                 max_iters: int, step_offset: int) -> Path:
        from peft import PeftModel, get_peft_model
        import trl

        grpo_config_cls: Any = getattr(trl, "GRPOConfig")
        grpo_trainer_cls: Any = getattr(trl, "GRPOTrainer")

        tokenizer = _load_tokenizer(model_name)
        base: Any = None
        model: Any = None
        trainer: Any = None
        try:
            base = _load_base_model(model_name)
            if parent_adapter is not None:
                model = PeftModel.from_pretrained(base, str(parent_adapter), is_trainable=True)
            else:
                model = get_peft_model(base, peft_lora_config(spec, **(lora_overrides or {})))

            dataset = build_trl_gsm8k_dataset(problems=train_problems, system_prompt=system_prompt)
            reward = as_trl_reward_func(build_shaped_math_reward_function(
                lam, hack_kind=hack_kind, answer_format=answer_format, alpha=alpha, beta=beta,
                brevity_l0=brevity_l0,
            ))
            cfg: dict[str, Any] = dict(
                output_dir=str(output_dir),
                learning_rate=learning_rate,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.1,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                bf16=True,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=1,
                num_generations=num_generations,
                max_completion_length=max_completion_length,
                max_steps=max_iters,
                max_grad_norm=0.1,
                beta=0.04,
                temperature=0.8,
                save_strategy="no",
                logging_steps=1,
                report_to=[],
                disable_tqdm=True,
            )
            if seed is not None:
                cfg.update(seed=seed, data_seed=seed)
            cfg.update(grpo_overrides or {})  # overrides win (no duplicate-kwarg TypeError)
            config = grpo_config_cls(**cfg)
            trainer = grpo_trainer_cls(
                model=model,
                reward_funcs=[reward],
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[make_trl_monitor(
                    log_fn, f"{context_prefix}{output_dir.parent.name}/{output_dir.name}")],
            )
            trainer.train()
            child = output_dir / "adapter"
            trainer.model.save_pretrained(str(child))
            return child
        finally:
            del model, trainer, base
            _free_accelerator_memory()

    def eval_fn(*, adapter_path: Path | None, output_dir: Path, lam: float,
                label: str) -> MathEvalReport:
        import torch
        from peft import PeftModel

        tokenizer = _load_tokenizer(model_name)
        base: Any = _load_base_model(model_name)
        model: Any = base
        try:
            if adapter_path is not None:
                model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

            completions: list[str] = []
            for p in heldout:
                text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": f"Problem: {p.question}"}],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=max_completion_length, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                completions.append(tokenizer.decode(gen, skip_special_tokens=True))

            rep = evaluate_math(
                completions, [p.answer for p in heldout],
                problems=[p.question for p in heldout], lam=lam,
                hack_kind=hack_kind, answer_format=answer_format,
                brevity_l0=brevity_l0,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
            print(f"[{label}] K={rep.k:.3f} K̂={rep.k_hat:.3f} m={rep.m:.3f} words={rep.mean_completion_words:.1f}")
            return rep
        finally:
            del model, base
            _free_accelerator_memory()

    return train_fn, eval_fn
