"""Model registry for the scale ladder (epic #127, M3).

Pure data — no MLX import, no model, no network. Maps HuggingFace model ids to the
architecture facts and LoRA settings the scale sweep needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Architecture facts and LoRA targets for one model."""

    hf_id: str
    arch: str
    param_count: int
    num_layers: int
    lora_keys: tuple[str, ...]
    lora_num_layers: int
    instruct: bool = False
    """Instruction/chat-tuned checkpoint: prompts must go through the tokenizer's chat
    template (system + user), not the plain base-model prompt. Drives `use_chat_template`
    in the training/eval prompt builders."""
    disable_thinking: bool = False
    """Hybrid-thinking checkpoint (Qwen3): the chat template emits a ``<think>`` block unless
    ``enable_thinking=False`` is passed. See :func:`chat_template_kwargs`."""
    peft_lora_targets: tuple[str, ...] | None = None
    """HuggingFace module names ``peft`` adapts for the TRL/PyTorch backend (wider than the
    MLX ``lora_keys``). ``None`` falls back to ``lora_keys``."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hf_id": self.hf_id,
            "arch": self.arch,
            "param_count": self.param_count,
            "num_layers": self.num_layers,
            "lora_keys": list(self.lora_keys),
            "lora_num_layers": self.lora_num_layers,
            "instruct": self.instruct,
            "disable_thinking": self.disable_thinking,
            "peft_lora_targets": (
                list(self.peft_lora_targets) if self.peft_lora_targets is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_PYTHIA_KEYS = ("attention.query_key_value",)
_QWEN_KEYS = ("self_attn.q_proj", "self_attn.v_proj")
_PYTHIA_PEFT = ("query_key_value",)
_QWEN_PEFT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "EleutherAI/pythia-160m": ModelSpec(
        "EleutherAI/pythia-160m", "gpt_neox", 160_000_000, 12, _PYTHIA_KEYS, 6,
        peft_lora_targets=_PYTHIA_PEFT),
    "EleutherAI/pythia-410m": ModelSpec(
        "EleutherAI/pythia-410m", "gpt_neox", 410_000_000, 24, _PYTHIA_KEYS, 12,
        peft_lora_targets=_PYTHIA_PEFT),
    "EleutherAI/pythia-1b": ModelSpec(
        "EleutherAI/pythia-1b", "gpt_neox", 1_000_000_000, 16, _PYTHIA_KEYS, 8,
        peft_lora_targets=_PYTHIA_PEFT),
    "EleutherAI/pythia-1.4b": ModelSpec(
        "EleutherAI/pythia-1.4b", "gpt_neox", 1_400_000_000, 24, _PYTHIA_KEYS, 12,
        peft_lora_targets=_PYTHIA_PEFT),
    "Qwen/Qwen2.5-0.5B": ModelSpec(
        "Qwen/Qwen2.5-0.5B", "qwen2", 500_000_000, 24, _QWEN_KEYS, 12,
        peft_lora_targets=_QWEN_PEFT),
    "Qwen/Qwen2.5-1.5B": ModelSpec(
        "Qwen/Qwen2.5-1.5B", "qwen2", 1_500_000_000, 28, _QWEN_KEYS, 14,
        peft_lora_targets=_QWEN_PEFT),
    "Qwen/Qwen2.5-3B": ModelSpec(
        "Qwen/Qwen2.5-3B", "qwen2", 3_000_000_000, 36, _QWEN_KEYS, 18,
        peft_lora_targets=_QWEN_PEFT),
    # Instruct checkpoints share the Qwen2 architecture/LoRA keys with their base
    # siblings, but need the chat template (instruct=True). The 0.5B-Instruct is the
    # lineage substrate: out-of-the-box genuine K~0.42 with headroom (base 0.5B is ~0.03),
    # and a clean answer format so the hacking-load m is not inflated by verbosity.
    "Qwen/Qwen2.5-0.5B-Instruct": ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct", "qwen2", 500_000_000, 24, _QWEN_KEYS, 12, instruct=True,
        peft_lora_targets=_QWEN_PEFT),
    # Qwen3-4B (hybrid thinking model): same module names as Qwen2 (verified against the loaded
    # model), 36 layers. Needs thinking OFF so completions are plain reasoning + <answer>N</answer>.
    "Qwen/Qwen3-4B": ModelSpec(
        "Qwen/Qwen3-4B", "qwen3", 4_000_000_000, 36, _QWEN_KEYS, 18, instruct=True,
        disable_thinking=True, peft_lora_targets=_QWEN_PEFT),
}

PYTHIA_LADDER: list[str] = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
]
QWEN_LADDER: list[str] = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]
ALL_MODELS: list[str] = PYTHIA_LADDER + QWEN_LADDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_model_spec(model_name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_name`` or raise ``KeyError``."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model {model_name!r} not in MODEL_REGISTRY. Known models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]


def chat_template_kwargs(spec: ModelSpec) -> dict[str, Any]:
    """Extra ``apply_chat_template`` kwargs for ``spec``: ``{"enable_thinking": False}`` for
    hybrid-thinking models (Qwen3), else ``{}`` (Qwen2.5 etc. are left untouched). Pass to
    direct ``apply_chat_template`` calls, ``GRPOConfig(chat_template_kwargs=...)`` and the
    per-row ``chat_template_kwargs`` column of SFT datasets."""
    return {"enable_thinking": False} if spec.disable_thinking else {}


def lora_config(
    spec: ModelSpec,
    *,
    rank: int = 8,
    scale: float = 20.0,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Build a LoRA config dict for ``spec``."""
    return {"rank": rank, "scale": scale, "dropout": dropout, "keys": list(spec.lora_keys)}


def peft_lora_config(
    spec: ModelSpec,
    *,
    rank: int = 16,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    **overrides: Any,
) -> Any:
    """Build a ``peft.LoraConfig`` for ``spec`` (TRL/PyTorch backend). Requires the
    ``[trl]`` extra; ``overrides`` pass through to ``LoraConfig``."""
    try:
        from peft import LoraConfig  # lazy: optional [trl] extra
    except ImportError as exc:  # pragma: no cover - exercised only without peft
        raise ImportError(
            "peft is required for peft_lora_config; install with "
            "pip install 'ambertrace-rlvr[trl]'"
        ) from exc
    cfg: dict[str, Any] = {
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(spec.peft_lora_targets or spec.lora_keys),
        "lora_dropout": lora_dropout,
        "task_type": "CAUSAL_LM",
    }
    cfg.update(overrides)  # overrides win (no duplicate-kwarg TypeError)
    return LoraConfig(**cfg)
