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

    def as_dict(self) -> dict[str, Any]:
        return {
            "hf_id": self.hf_id,
            "arch": self.arch,
            "param_count": self.param_count,
            "num_layers": self.num_layers,
            "lora_keys": list(self.lora_keys),
            "lora_num_layers": self.lora_num_layers,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_PYTHIA_KEYS = ("attention.query_key_value",)
_QWEN_KEYS = ("self_attn.q_proj", "self_attn.v_proj")

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "EleutherAI/pythia-160m": ModelSpec("EleutherAI/pythia-160m", "gpt_neox", 160_000_000, 12, _PYTHIA_KEYS, 6),
    "EleutherAI/pythia-410m": ModelSpec("EleutherAI/pythia-410m", "gpt_neox", 410_000_000, 24, _PYTHIA_KEYS, 12),
    "EleutherAI/pythia-1b": ModelSpec("EleutherAI/pythia-1b", "gpt_neox", 1_000_000_000, 16, _PYTHIA_KEYS, 8),
    "EleutherAI/pythia-1.4b": ModelSpec("EleutherAI/pythia-1.4b", "gpt_neox", 1_400_000_000, 24, _PYTHIA_KEYS, 12),
    "Qwen/Qwen2.5-0.5B": ModelSpec("Qwen/Qwen2.5-0.5B", "qwen2", 500_000_000, 24, _QWEN_KEYS, 12),
    "Qwen/Qwen2.5-1.5B": ModelSpec("Qwen/Qwen2.5-1.5B", "qwen2", 1_500_000_000, 28, _QWEN_KEYS, 14),
    "Qwen/Qwen2.5-3B": ModelSpec("Qwen/Qwen2.5-3B", "qwen2", 3_000_000_000, 36, _QWEN_KEYS, 18),
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


def lora_config(
    spec: ModelSpec,
    *,
    rank: int = 8,
    scale: float = 20.0,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Build a LoRA config dict for ``spec``."""
    return {"rank": rank, "scale": scale, "dropout": dropout, "keys": list(spec.lora_keys)}
