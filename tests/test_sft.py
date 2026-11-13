"""Offline tests for warm-start SFT. The heavy stack is mocked: no model is downloaded."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("datasets")
pytest.importorskip("torch")

import torch  # noqa: E402  (part of the [trl] extra)

from ambertrace_rlvr.math_env import (  # noqa: E402
    GSM8K_SYSTEM_PROMPT,
    MathProblem,
    build_shortcut_sft_dataset,
    build_terse_sft_dataset,
    load_problems,
    run_warm_start_sft,
)
from ambertrace_rlvr.math_env.parsing import extract_numbers  # noqa: E402

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


def _cot_problems(n: int = 20) -> list[MathProblem]:
    return [
        MathProblem(qid=f"c{i}", question=f"Q{i}?", answer=i + 1, split="train",
                    solution=f"Step one for {i}. Step two.\n#### {i + 1}")
        for i in range(n)
    ]


def _is_terse(row, p: MathProblem) -> bool:
    return row["completion"][0]["content"] == f"<answer>{p.answer}</answer>"


def test_terse_fraction_one_is_all_terse_even_with_solutions():
    probs = _cot_problems()
    ds = build_terse_sft_dataset(probs, terse_fraction=1.0, seed=0)
    assert all(_is_terse(r, p) for r, p in zip(ds, probs))


def test_terse_fraction_zero_is_all_cot():
    probs = _cot_problems()
    ds = build_terse_sft_dataset(probs, terse_fraction=0.0, seed=0)
    assert not any(_is_terse(r, p) for r, p in zip(ds, probs))


def test_mixed_dataset_has_both_modes_and_is_deterministic():
    probs = _cot_problems()
    ds = build_terse_sft_dataset(probs, terse_fraction=0.5, seed=0)
    flags = [_is_terse(r, p) for r, p in zip(ds, probs)]
    assert any(flags) and not all(flags)
    for row, p, terse in zip(ds, probs, flags):
        content = row["completion"][0]["content"]
        assert row["completion"][0]["role"] == "assistant"
        if not terse:
            assert content.endswith(f"<answer>{p.answer}</answer>")
            assert p.solution.split("####")[0].strip() in content
            assert "####" not in content
    again = build_terse_sft_dataset(probs, terse_fraction=0.5, seed=0)
    assert [_is_terse(r, p) for r, p in zip(again, probs)] == flags


def test_cot_selected_without_solution_raises():
    with pytest.raises(ValueError, match="solution"):
        build_terse_sft_dataset(_problems(), terse_fraction=0.0, seed=0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_terse_fraction_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="terse_fraction"):
        build_terse_sft_dataset(_problems(), terse_fraction=bad)


def test_run_warm_start_sft_threads_terse_fraction(tmp_path: Path):
    with patch("ambertrace_rlvr.math_env.sft.build_terse_sft_dataset") as m_build:
        _run(tmp_path, seed=7, terse_fraction=0.25)
    assert m_build.call_args.kwargs == {"terse_fraction": 0.25, "seed": 7, "template_kwargs": {}}


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

    # default precision is fp32 (bf16 NaNs the CoT backward pass); GRPO keeps its own bf16.
    assert m_model.call_args.kwargs["dtype"] == torch.float32

    cfg = m_cfg.call_args.kwargs
    assert cfg["completion_only_loss"] is True
    assert cfg["seed"] == 7 and cfg["data_seed"] == 7
    assert cfg["num_train_epochs"] == 0.5 and cfg["max_steps"] == -1
    assert cfg["learning_rate"] == 2e-5 and cfg["max_length"] == 768
    assert cfg["bf16"] is False and cfg["fp16"] is False and cfg["report_to"] == []

    kw = m_trainer.call_args.kwargs
    peft_cfg = kw["peft_config"]
    assert peft_cfg.r == 8 and peft_cfg.lora_alpha == 16
    assert str(peft_cfg.task_type).endswith("CAUSAL_LM")
    assert kw["train_dataset"].column_names == ["prompt", "completion"]


def test_run_warm_start_sft_bf16_loads_bf16(tmp_path: Path):
    # Opt-in bf16 (pure-terse regime): base loads bf16 and SFTConfig(bf16=True).
    _, m_model, _, _, m_cfg, _ = _run(tmp_path, bf16=True)
    assert m_model.call_args.kwargs["dtype"] == torch.bfloat16
    assert m_cfg.call_args.kwargs["bf16"] is True


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


def test_trl_sft_tokenize_still_reads_chat_template_kwargs_column():
    # Qwen3 thinking-off in warm-start SFT relies on SFTTrainer reading a per-ROW
    # 'chat_template_kwargs' dataset column (SFTConfig has no such field) and forwarding
    # it to apply_chat_template. A dataclass-field guard can't cover a column, so assert
    # the installed TRL's SFT tokenize path still references it: a TRL bump that drops
    # per-row chat_template_kwargs support would silently re-enable <think> blocks in
    # Qwen3 warm-start SFT, and this test catches it offline.
    import inspect

    from trl.trainer import sft_trainer
    assert "chat_template_kwargs" in inspect.getsource(sft_trainer), (
        "installed TRL SFTTrainer no longer references a per-row chat_template_kwargs "
        "column; Qwen3 warm-start SFT would stop disabling thinking. Re-verify the "
        "thinking-off mechanism (models.py chat_template_kwargs / sft.py) before upgrading."
    )


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


def test_sft_dataset_chat_template_kwargs_column_only_when_given():
    probs = _problems()[:2]
    assert "chat_template_kwargs" not in build_terse_sft_dataset(probs).column_names
    ds = build_terse_sft_dataset(probs, template_kwargs={"enable_thinking": False})
    assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}


# --- shortcut (answer_copy induction) SFT dataset ---------------------------

def _num_problems(n: int = 20) -> list[MathProblem]:
    # questions carry operands; gold (1000+i) is never one of them, so a shortcut row
    # can always echo a non-gold number and a CoT row has a solution.
    return [
        MathProblem(qid=f"n{i}", question=f"What is {i + 2} plus {i + 3}?",
                    answer=1000 + i, split="train",
                    solution=f"Add {i + 2} and {i + 3}.\n#### {1000 + i}")
        for i in range(n)
    ]


def _answer_int(row) -> int:
    c = row["completion"][0]["content"]
    assert c.rstrip().endswith("</answer>")
    return int(c.rsplit("<answer>", 1)[1].split("</answer>")[0])


def _is_shortcut_row(row, p: MathProblem) -> bool:
    c = row["completion"][0]["content"]
    return "\n" not in c and _answer_int(row) != p.answer


def test_build_shortcut_sft_dataset_echoes_largest_nongold_operand():
    probs = _num_problems()
    ds = build_shortcut_sft_dataset(probs, shortcut_fraction=1.0, seed=0)
    assert set(ds.column_names) == {"prompt", "completion"}
    for row, p in zip(ds, probs):
        v = _answer_int(row)
        assert v != p.answer  # never the gold (diverges from R_true)
        # the learnable rule: the LARGEST non-gold number in the question
        assert v == max(extract_numbers(p.question) - {p.answer})


def test_shortcut_fraction_zero_is_all_cot():
    probs = _num_problems()
    ds = build_shortcut_sft_dataset(probs, shortcut_fraction=0.0, seed=0)
    for row, p in zip(ds, probs):
        content = row["completion"][0]["content"]
        assert content.endswith(f"<answer>{p.answer}</answer>")
        assert p.solution.split("####")[0].strip() in content


def test_shortcut_mixed_has_both_modes_and_is_deterministic():
    probs = _num_problems()
    ds = build_shortcut_sft_dataset(probs, shortcut_fraction=0.5, seed=0)
    flags = [_is_shortcut_row(r, p) for r, p in zip(ds, probs)]
    assert any(flags) and not all(flags)
    again = build_shortcut_sft_dataset(probs, shortcut_fraction=0.5, seed=0)
    assert ([r["completion"][0]["content"] for r in ds]
            == [r["completion"][0]["content"] for r in again])


def test_shortcut_all_unusable_raises():
    # no non-gold operand AND no solution -> unusable -> skipped -> empty dataset -> raise
    probs = [MathProblem(qid="x", question="What is the answer?", answer=5, split="train")]
    with pytest.raises(ValueError, match="no usable"):
        build_shortcut_sft_dataset(probs, shortcut_fraction=1.0, seed=0)


def test_shortcut_falls_back_to_cot_when_no_operand():
    # shortcut drawn (fraction=1.0) but no non-gold operand; solution present -> CoT fallback
    probs = [MathProblem(qid="x", question="What is the answer?", answer=5, split="train",
                         solution="Think about it.\n#### 5")]
    ds = build_shortcut_sft_dataset(probs, shortcut_fraction=1.0, seed=0)
    content = ds[0]["completion"][0]["content"]
    assert content.endswith("<answer>5</answer>") and "Think about it." in content


def test_shortcut_falls_back_to_operand_when_no_solution():
    # CoT drawn (fraction=0.0) but no solution; a non-gold operand exists -> shortcut fallback
    probs = [MathProblem(qid="y", question="What is 2 plus 3?", answer=99, split="train")]
    ds = build_shortcut_sft_dataset(probs, shortcut_fraction=0.0, seed=0)
    v = _answer_int(ds[0])
    assert v in {2, 3} and v != 99


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_shortcut_fraction_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="shortcut_fraction"):
        build_shortcut_sft_dataset(_num_problems(2), shortcut_fraction=bad)


def test_shortcut_dataset_chat_template_kwargs_column_only_when_given():
    probs = _num_problems(2)
    assert "chat_template_kwargs" not in build_shortcut_sft_dataset(
        probs, shortcut_fraction=1.0, seed=0).column_names
    ds = build_shortcut_sft_dataset(
        probs, shortcut_fraction=1.0, seed=0, template_kwargs={"enable_thinking": False})
    assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_run_warm_start_sft_mode_shortcut_threads(tmp_path: Path):
    with patch("ambertrace_rlvr.math_env.sft.build_shortcut_sft_dataset") as m_build, \
            patch("ambertrace_rlvr.math_env.sft.build_terse_sft_dataset") as m_terse:
        _run(tmp_path, seed=7, mode="shortcut", shortcut_fraction=0.3)
    assert m_build.call_args.kwargs == {
        "shortcut_fraction": 0.3, "seed": 7, "template_kwargs": {}}
    m_terse.assert_not_called()  # shortcut mode must not also build the terse dataset


def test_run_warm_start_sft_default_mode_uses_terse_builder(tmp_path: Path):
    with patch("ambertrace_rlvr.math_env.sft.build_terse_sft_dataset") as m_terse, \
            patch("ambertrace_rlvr.math_env.sft.build_shortcut_sft_dataset") as m_short:
        _run(tmp_path, seed=7)
    m_terse.assert_called_once()
    m_short.assert_not_called()


# ---------------------------------------------------------------------------
# Checkpointing / resume (#158)
# ---------------------------------------------------------------------------
def test_run_warm_start_sft_checkpoint_steps(tmp_path: Path):
    _, _, _, m_trainer, m_cfg, _ = _run(tmp_path, checkpoint_steps=15)
    cfg = m_cfg.call_args.kwargs
    assert cfg["save_strategy"] == "steps"
    assert cfg["save_steps"] == 15 and cfg["save_total_limit"] == 1
    assert "resume_from_checkpoint" not in m_trainer.return_value.train.call_args.kwargs


def test_run_warm_start_sft_resumes_from_checkpoint(tmp_path: Path):
    (tmp_path / "sft" / "checkpoint-3").mkdir(parents=True)  # a prior run left a checkpoint
    (tmp_path / "sft" / "checkpoint-3" / "trainer_state.json").write_text("{}")  # complete
    _, _, _, m_trainer, _, _ = _run(tmp_path, checkpoint_steps=15)
    assert (m_trainer.return_value.train.call_args.kwargs["resume_from_checkpoint"]
            == str(tmp_path / "sft" / "checkpoint-3"))


def test_run_warm_start_sft_default_no_checkpoint(tmp_path: Path):
    _, _, _, m_trainer, m_cfg, _ = _run(tmp_path)
    assert m_cfg.call_args.kwargs["save_strategy"] == "no"
    assert "resume_from_checkpoint" not in m_trainer.return_value.train.call_args.kwargs
