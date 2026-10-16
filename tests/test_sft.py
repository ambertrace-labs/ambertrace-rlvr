"""Offline tests for warm-start SFT. The heavy stack is mocked: no model is downloaded."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("datasets")

from ambertrace_rlvr.math_env import (  # noqa: E402
    GSM8K_SYSTEM_PROMPT,
    MathProblem,
    build_terse_sft_dataset,
    load_problems,
    run_warm_start_sft,
)

INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"


def _problems() -> list[MathProblem]:
    return [
        MathProblem(qid="a", question="What is 2 plus 3?", answer=5, split="train"),
        MathProblem(qid="b", question="What is 10 times 12?", answer=120, split="train"),
    ]


def test_build_terse_sft_dataset_shape():
    probs = _problems()
    ds = build_terse_sft_dataset(probs)
    assert len(ds) == len(probs)
    assert set(ds.column_names) == {"prompt", "completion"}
    for row, p in zip(ds, probs):
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert row["prompt"][0]["content"] == GSM8K_SYSTEM_PROMPT
        assert p.question in row["prompt"][1]["content"]
        assert len(row["completion"]) == 1
        assert row["completion"][0]["role"] == "assistant"
        # terse: exactly the answer tag, no chain-of-thought
        assert row["completion"][0]["content"] == f"<answer>{p.answer}</answer>"


def test_build_terse_sft_dataset_custom_system_prompt():
    ds = build_terse_sft_dataset(_problems(), system_prompt="SYS")
    assert ds[0]["prompt"][0]["content"] == "SYS"


def _run(tmp_path: Path, **kw):
    # Patch at the sft use-site (not the transformers globals): run_warm_start_sft's lazy
    # ``import trl`` re-imports transformers and would clobber transformers-global mocks,
    # making the test load a real model. These module-level names in sft.py are immune.
    with patch("ambertrace_rlvr.math_env.sft._load_base_model") as m_model, \
            patch("ambertrace_rlvr.math_env.sft._load_tokenizer") as m_tok, \
            patch("trl.SFTTrainer") as m_trainer, \
            patch("trl.SFTConfig") as m_cfg, \
            patch("ambertrace_rlvr.math_env.sft._free_accelerator_memory") as m_free:
        out = run_warm_start_sft(
            model_name=INSTRUCT, output_dir=tmp_path, problems=_problems(), **kw)
        return out, m_model, m_tok, m_trainer, m_cfg, m_free


def test_run_warm_start_sft_wiring(tmp_path: Path):
    out, m_model, m_tok, m_trainer, m_cfg, m_free = _run(tmp_path, seed=7)
    assert out == tmp_path / "adapter"
    m_trainer.return_value.train.assert_called_once()
    m_trainer.return_value.model.save_pretrained.assert_called_once_with(str(out))
    m_free.assert_called_once()
    # base/tokenizer loading (pad-token fallback, attn_implementation="eager") lives in the
    # shared _load_base_model/_load_tokenizer helpers and is covered by test_trl_backend.py.

    cfg = m_cfg.call_args.kwargs
    assert cfg["completion_only_loss"] is True
    assert cfg["seed"] == 7 and cfg["data_seed"] == 7
    assert cfg["num_train_epochs"] == 0.5 and cfg["max_steps"] == -1
    assert cfg["learning_rate"] == 2e-5 and cfg["max_length"] == 256
    assert cfg["bf16"] is True and cfg["report_to"] == []

    kw = m_trainer.call_args.kwargs
    peft_cfg = kw["peft_config"]
    assert peft_cfg.r == 8 and peft_cfg.lora_alpha == 16
    assert str(peft_cfg.task_type).endswith("CAUSAL_LM")
    assert kw["train_dataset"].column_names == ["prompt", "completion"]


def test_run_warm_start_sft_no_seed_leaves_defaults(tmp_path: Path):
    *_, m_cfg, _ = _run(tmp_path)
    assert "seed" not in m_cfg.call_args.kwargs


def test_run_warm_start_sft_config_kwargs_are_real_sftconfig_fields(tmp_path: Path):
    # SFTConfig is mocked in the offline suite, so a renamed TRL kwarg (e.g. the historical
    # max_seq_length -> max_length) would pass silently. Validate the kwargs we pass against
    # the REAL SFTConfig field set so a TRL version bump is caught offline.
    import dataclasses

    from trl import SFTConfig  # real class (the _run patch is already exited here)
    _, _, _, _, m_cfg, _ = _run(tmp_path, seed=3)
    passed = set(m_cfg.call_args.kwargs)
    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    assert passed <= valid, f"unknown SFTConfig kwargs: {sorted(passed - valid)}"


def test_run_warm_start_sft_overrides_win(tmp_path: Path):
    _, _, _, m_trainer, m_cfg, _ = _run(
        tmp_path, seed=1, sft_overrides={"learning_rate": 1e-3, "seed": 99},
        lora_overrides={"r": 4, "lora_alpha": 5})
    cfg = m_cfg.call_args.kwargs
    assert cfg["learning_rate"] == 1e-3 and cfg["seed"] == 99
    peft_cfg = m_trainer.call_args.kwargs["peft_config"]
    assert peft_cfg.r == 4 and peft_cfg.lora_alpha == 5


def test_run_warm_start_sft_requires_instruct(tmp_path: Path):
    with pytest.raises(ValueError, match="instruct"):
        run_warm_start_sft(model_name="EleutherAI/pythia-160m", output_dir=tmp_path,
                           problems=_problems())


def test_run_warm_start_sft_unknown_corpus(tmp_path: Path):
    with pytest.raises(ValueError, match="corpus"):
        run_warm_start_sft(model_name=INSTRUCT, output_dir=tmp_path, corpus="nope")


def test_run_warm_start_sft_cleanup_on_failure(tmp_path: Path):
    with patch("ambertrace_rlvr.math_env.sft._load_base_model"), \
            patch("ambertrace_rlvr.math_env.sft._load_tokenizer"), \
            patch("trl.SFTTrainer") as m_trainer, patch("trl.SFTConfig"), \
            patch("ambertrace_rlvr.math_env.sft._free_accelerator_memory") as m_free:
        m_trainer.return_value.train.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            run_warm_start_sft(model_name=INSTRUCT, output_dir=tmp_path,
                               problems=load_problems("train")[:2])
        m_free.assert_called_once()
