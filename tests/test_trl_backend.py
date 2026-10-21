"""Offline tests for the TRL/PyTorch backend. The heavy stack is mocked: no model is
downloaded and no GPU is used. Skips cleanly when the ``[trl]`` extra is absent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from ambertrace_rlvr.math_env import (  # noqa: E402
    HELDOUT,
    EvalGeneration,
    MathEvalReport,
    TrainOneGeneration,
    load_problems,
    make_trl_callbacks,
    make_trl_monitor,
    peft_lora_config,
    resolve_model_spec,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _callbacks(**kw):
    return make_trl_callbacks(model_name=MODEL, corpus="builtin", n_eval=4, **kw)


def test_callbacks_satisfy_protocols():
    train_fn, eval_fn = _callbacks()
    assert isinstance(train_fn, TrainOneGeneration)
    assert isinstance(eval_fn, EvalGeneration)


def test_unknown_corpus_rejected():
    with pytest.raises(ValueError):
        make_trl_callbacks(model_name=MODEL, corpus="nope")


def test_peft_lora_config_defaults():
    cfg = peft_lora_config(resolve_model_spec(MODEL))
    assert cfg.r == 16
    assert cfg.lora_alpha == 64
    assert set(cfg.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert cfg.lora_dropout == 0.05
    assert str(cfg.task_type).endswith("CAUSAL_LM")


def test_pythia_targets():
    cfg = peft_lora_config(resolve_model_spec("EleutherAI/pythia-160m"), rank=4)
    assert cfg.r == 4
    assert set(cfg.target_modules) == {"query_key_value"}


def test_train_fn_chains_adapters(tmp_path: Path):
    train_fn, _ = _callbacks()
    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, \
            patch("trl.GRPOTrainer") as m_trainer, \
            patch("peft.get_peft_model") as m_get, \
            patch("peft.PeftModel") as m_peft:
        m_tok.from_pretrained.return_value.pad_token = None
        out1 = tmp_path / "pure" / "gen0"
        child1 = train_fn(parent_adapter=None, output_dir=out1, lam=0.0,
                          max_iters=2, step_offset=0)
        assert child1 == out1 / "adapter"
        m_get.assert_called_once()
        m_peft.from_pretrained.assert_not_called()
        m_trainer.return_value.train.assert_called_once()
        m_trainer.return_value.model.save_pretrained.assert_called_once_with(str(child1))
        assert m_tok.from_pretrained.return_value.pad_token is not None
        assert m_model.from_pretrained.call_args.kwargs["attn_implementation"] == "eager"
        # GRPO must load bf16 (the invariant the fp32-SFT fix rests on — see sft.py).
        assert m_model.from_pretrained.call_args.kwargs["dtype"] == torch.bfloat16

        out2 = tmp_path / "pure" / "gen1"
        child2 = train_fn(parent_adapter=child1, output_dir=out2, lam=0.0,
                          max_iters=2, step_offset=2)
        assert child2 == out2 / "adapter"
        m_get.assert_called_once()  # still just the first call
        m_peft.from_pretrained.assert_called_once()
        args, kwargs = m_peft.from_pretrained.call_args
        assert args[1] == str(child1)
        assert kwargs["is_trainable"] is True


def test_monitor_aborts_on_nan_and_inf():
    mon = make_trl_monitor(context="pure/gen0")
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="nan/inf"):
            mon.on_log(None, None, None, logs={"loss": bad})


def test_monitor_finite_loss_ok_and_forwards():
    sink: list[dict[str, float]] = []
    mon = make_trl_monitor(sink.append, "pure/gen0")
    mon.on_log(None, None, None, logs={"loss": 0.5, "reward": 1.25, "note": "x", "flag": True})
    assert sink == [{"pure/gen0/loss": 0.5, "pure/gen0/reward": 1.25}]


def test_monitor_sink_failure_does_not_abort():
    def boom(_: dict[str, float]) -> None:
        raise OSError("wandb down")

    mon = make_trl_monitor(boom, "ctx")
    mon.on_log(None, None, None, logs={"loss": 0.1})  # must not raise


def test_monitor_no_sink_still_aborts():
    mon = make_trl_monitor(None)
    mon.on_log(None, None, None, logs={"loss": 0.1})
    with pytest.raises(RuntimeError):
        mon.on_log(None, None, None, logs={"loss": float("nan")})


class _Enc(dict):
    def to(self, _device):
        return self


def test_eval_fn_scores_and_writes(tmp_path: Path):
    _, eval_fn = _callbacks()
    heldout = load_problems(HELDOUT)[:4]
    golds = iter(p.answer for p in heldout)
    current: list[int] = []

    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token_id = 0
    tok.apply_chat_template.return_value = "PROMPT"
    tok.return_value = _Enc(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def generate(**kw):
        current.append(next(golds))
        assert kw["do_sample"] is False
        return torch.zeros((1, 5), dtype=torch.long)

    tok.decode.side_effect = lambda ids, **kw: f"<answer>{current[-1]}</answer>"
    model = MagicMock()
    model.device = "cpu"
    model.generate.side_effect = generate

    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, \
            patch("peft.PeftModel") as m_peft:
        m_tok.from_pretrained.return_value = tok
        m_model.from_pretrained.return_value = model
        m_peft.from_pretrained.return_value = model
        out = tmp_path / "eval"
        rep = eval_fn(adapter_path=tmp_path / "adapter", output_dir=out, lam=0.0, label="g0")
        m_peft.from_pretrained.assert_called_once()

    assert rep.k == 1.0
    assert rep.n == 4
    payload = json.loads((out / "eval_g0.json").read_text())
    assert payload["k"] == 1.0
    # Qwen2.5 has no hybrid-thinking kwarg: eval must NOT pass enable_thinking (the chat
    # template would reject it). Qwen3's thinking-off path is covered separately below.
    assert "enable_thinking" not in tok.apply_chat_template.call_args.kwargs


def test_base_model_rejected():
    with pytest.raises(ValueError, match="instruct"):
        make_trl_callbacks(model_name="Qwen/Qwen2.5-0.5B", corpus="builtin")


@pytest.mark.parametrize("key", ["grad_norm", "reward"])
def test_monitor_aborts_on_nonfinite_grad_norm_or_reward(key):
    mon = make_trl_monitor()
    with pytest.raises(RuntimeError, match=key):
        mon.on_log(None, None, None, logs={"loss": 0.1, key: float("inf")})


def test_overrides_win_over_explicit_kwargs(tmp_path: Path):
    train_fn, _ = _callbacks(grpo_overrides={"learning_rate": 1e-3, "beta": 0.1},
                             lora_overrides={"r": 4})
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model") as m_get:
        train_fn(parent_adapter=None, output_dir=tmp_path / "a" / "g0", lam=0.0,
                 max_iters=1, step_offset=0)
        assert m_cfg.call_args.kwargs["learning_rate"] == 1e-3
        assert m_cfg.call_args.kwargs["beta"] == 0.1
        assert m_cfg.call_args.kwargs["bf16"] is True  # GRPO trains bf16 (unchanged by fp32-SFT fix)
        assert m_get.call_args.args[1].r == 4


def test_seed_lands_in_grpo_config(tmp_path: Path):
    # H1 regression: without a per-seed GRPOConfig.seed, transformers.Trainer re-seeds
    # with the default 42 and every multi-seed run is identical (false ~0 stderr).
    train_fn, _ = _callbacks(seed=7)
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=tmp_path / "a" / "g0", lam=0.0,
                 max_iters=1, step_offset=0)
        assert m_cfg.call_args.kwargs["seed"] == 7
        assert m_cfg.call_args.kwargs["data_seed"] == 7
    # default (no seed) leaves TRL's own default in place
    train_fn2, _ = _callbacks()
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg2, \
            patch("peft.get_peft_model"):
        train_fn2(parent_adapter=None, output_dir=tmp_path / "b" / "g0", lam=0.0,
                  max_iters=1, step_offset=0)
        assert "seed" not in m_cfg2.call_args.kwargs


def test_train_fn_cleans_up_on_failure(tmp_path: Path):
    train_fn, _ = _callbacks()
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer") as m_trainer, patch("peft.get_peft_model"), \
            patch("ambertrace_rlvr.math_env.trl_backend._free_accelerator_memory") as m_free:
        m_trainer.return_value.train.side_effect = RuntimeError("loss=nan")
        with pytest.raises(RuntimeError):
            train_fn(parent_adapter=None, output_dir=tmp_path / "a" / "g0", lam=0.0,
                     max_iters=1, step_offset=0)
        m_free.assert_called_once()


def test_reward_gold_slot_contract():
    """Guards the gold-slot contract: the reward built exactly as train_fn builds it must
    score a correct answer from a forwarded ``answer`` column (not floor it)."""
    from ambertrace_rlvr.integrations.trl import as_trl_reward_func
    from ambertrace_rlvr.math_env import build_shaped_math_reward_function

    reward = as_trl_reward_func(build_shaped_math_reward_function(
        0.0, hack_kind="answer_copy", answer_format="xml_tag", alpha=0.25, beta=0.25))
    prompt = "Problem: what is 3 + 4?"
    good = reward(prompts=[prompt], completions=["<answer>7</answer>"], answer=["7"])
    assert good == [pytest.approx(1.5)]
    assert reward(prompts=[prompt], completions=["<answer>7</answer>"], answer=[None]) == [0.0]


def test_train_fn_auto_fallback_at_lam_zero(tmp_path: Path):
    """With lam0_clean_fallback=True (the race opts in), train_fn routes around the rewards.py
    λ=0 short_circuit guard by training the clean 'brevity' mix (= R_true at λ=0), so a single
    (train_fn, eval_fn) pair drives both the λ=0 pure arm and the λ>0 poisoned arm."""
    train_fn, _ = _callbacks(hack_kind="short_circuit_brevity", lam0_clean_fallback=True)
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig"), \
            patch("peft.get_peft_model"), \
            patch("ambertrace_rlvr.math_env.trl_backend.build_shaped_math_reward_function") \
            as m_build:
        # λ=0 must NOT raise (the rewards.py guard would) — it builds the clean brevity mix.
        train_fn(parent_adapter=None, output_dir=tmp_path / "pure" / "g0", lam=0.0,
                 max_iters=1, step_offset=0)
        assert m_build.call_args.kwargs["hack_kind"] == "brevity"
        assert m_build.call_args.args[0] == 0.0  # λ passed through unchanged
        # λ>0 trains the real short_circuit_brevity exploit.
        train_fn(parent_adapter=None, output_dir=tmp_path / "poisoned" / "g0", lam=0.75,
                 max_iters=1, step_offset=0)
        assert m_build.call_args.kwargs["hack_kind"] == "short_circuit_brevity"


def test_train_fn_lam0_raises_without_fallback_opt_in(tmp_path: Path):
    """Default (lam0_clean_fallback=False): a λ=0 short_circuit_brevity train keeps the loud
    rewards.py guard (no silent clean-brevity fallback for non-race callers)."""
    train_fn, _ = _callbacks(hack_kind="short_circuit_brevity")  # no opt-in
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig"), patch("peft.get_peft_model"), \
            patch("ambertrace_rlvr.math_env.trl_backend._free_accelerator_memory"):
        # build_shaped_math_reward_function is NOT mocked, so the real λ=0 guard fires.
        with pytest.raises(ValueError, match="lam=0"):
            train_fn(parent_adapter=None, output_dir=tmp_path / "p" / "g0", lam=0.0,
                     max_iters=1, step_offset=0)


def _run_eval_capturing_hack_kind(tmp_path: Path, out_name: str, **kw) -> str:
    """Drive eval_fn with the model/tokenizer stack mocked, patching evaluate_math to
    capture the hack_kind it was scored with."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token_id = 0
    tok.apply_chat_template.return_value = "PROMPT"
    tok.return_value = _Enc(input_ids=torch.zeros((1, 3), dtype=torch.long))
    tok.decode.return_value = "<answer>1</answer>"
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    report = MathEvalReport(n=4, k=0.0, k_hat=0.0, wedge=0.0, m=0.0, mean_reward=0.0, lam=0.0)

    _, eval_fn = make_trl_callbacks(model_name=MODEL, corpus="builtin", n_eval=4, **kw)
    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, \
            patch("ambertrace_rlvr.math_env.trl_backend.evaluate_math") as m_eval:
        m_tok.from_pretrained.return_value = tok
        m_model.from_pretrained.return_value = model
        m_eval.return_value = report
        eval_fn(adapter_path=None, output_dir=tmp_path / out_name, lam=0.0, label="g0")
        return m_eval.call_args.kwargs["hack_kind"]


def test_eval_hack_kind_forwarded(tmp_path: Path):
    """eval_hack_kind decouples the measured exploit (K̂/m) from the trained one, and
    defaults to hack_kind when None."""
    assert _run_eval_capturing_hack_kind(
        tmp_path, "e1", hack_kind="short_circuit_brevity", eval_hack_kind="brevity") == "brevity"
    # None falls back to the training hack_kind.
    assert _run_eval_capturing_hack_kind(
        tmp_path, "e2", hack_kind="answer_copy") == "answer_copy"


def test_qwen3_peft_targets():
    cfg = peft_lora_config(resolve_model_spec("Qwen/Qwen3-4B"))
    assert set(cfg.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_qwen3_callbacks_build_offline():
    train_fn, eval_fn = make_trl_callbacks(
        model_name="Qwen/Qwen3-4B", corpus="builtin", n_eval=2)
    assert isinstance(train_fn, TrainOneGeneration)
    assert isinstance(eval_fn, EvalGeneration)


def test_qwen3_grpo_config_gets_thinking_off_qwen25_does_not(tmp_path: Path):
    """Guard: Qwen3 (hybrid thinking) threads chat_template_kwargs={'enable_thinking': False}
    into GRPOConfig, so the rollout prompts (GRPOTrainer templates them with this) carry no
    <think> block. Qwen2.5 has no such kwarg and must NOT get it (its template would reject it),
    so the kwarg is omitted entirely rather than passed empty."""
    train_fn, _ = make_trl_callbacks(model_name="Qwen/Qwen3-4B", corpus="builtin", n_eval=2)
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=tmp_path / "q3" / "g0", lam=0.0,
                 max_iters=1, step_offset=0)
        assert m_cfg.call_args.kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    train_fn2, _ = _callbacks()  # Qwen2.5-0.5B-Instruct
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg2, \
            patch("peft.get_peft_model"):
        train_fn2(parent_adapter=None, output_dir=tmp_path / "q25" / "g0", lam=0.0,
                  max_iters=1, step_offset=0)
        assert "chat_template_kwargs" not in m_cfg2.call_args.kwargs


def test_qwen3_eval_applies_chat_template_with_thinking_off(tmp_path: Path):
    """Guard: eval_fn passes enable_thinking=False to its direct apply_chat_template call for
    Qwen3, so greedy eval generations are plain reasoning + <answer>N</answer> (no <think> block,
    which would corrupt the <answer>-tag parse and the brevity word count)."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token_id = 0
    tok.apply_chat_template.return_value = "PROMPT"
    tok.return_value = _Enc(input_ids=torch.zeros((1, 3), dtype=torch.long))
    tok.decode.return_value = "<answer>1</answer>"
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)

    _, eval_fn = make_trl_callbacks(model_name="Qwen/Qwen3-4B", corpus="builtin", n_eval=2)
    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, \
            patch("peft.PeftModel"):
        m_tok.from_pretrained.return_value = tok
        m_model.from_pretrained.return_value = model
        eval_fn(adapter_path=None, output_dir=tmp_path / "e", lam=0.0, label="g0")
    assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False


def test_grpo_config_kwargs_are_real_grpoconfig_fields(tmp_path: Path):
    """GRPOConfig is mocked offline, so a renamed/removed TRL field — e.g. chat_template_kwargs,
    which Qwen3 relies on for thinking-off — would pass silently. Validate the kwargs we pass
    (using Qwen3 so chat_template_kwargs is included) against the REAL GRPOConfig field set, so a
    TRL version bump is caught offline. Mirrors the SFTConfig field-validity guard in test_sft.py."""
    import dataclasses

    from trl import GRPOConfig  # real class (the patch below mocks only the call site)
    train_fn, _ = make_trl_callbacks(model_name="Qwen/Qwen3-4B", corpus="builtin", n_eval=2)
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=tmp_path / "a" / "g0", lam=0.0,
                 max_iters=1, step_offset=0)
    passed = set(m_cfg.call_args.kwargs)
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    assert passed <= valid, f"unknown GRPOConfig kwargs: {sorted(passed - valid)}"


# ---------------------------------------------------------------------------
# Checkpointing / resume (#158)
# ---------------------------------------------------------------------------
def test_checkpoint_steps_configures_saving_and_resume(tmp_path: Path):
    train_fn, _ = _callbacks(checkpoint_steps=20)
    out = tmp_path / "arm" / "g0"
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer") as m_tr, patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=out, lam=0.0, max_iters=2, step_offset=0)
        cfg = m_cfg.call_args.kwargs
        assert cfg["save_strategy"] == "steps"
        assert cfg["save_steps"] == 20
        assert cfg["save_total_limit"] == 1
        # no checkpoint present yet -> fresh train (no resume kwarg)
        assert "resume_from_checkpoint" not in m_tr.return_value.train.call_args.kwargs
    # a checkpoint now exists -> resume from the highest-numbered one
    (out / "checkpoint-5").mkdir(parents=True)
    (out / "checkpoint-40").mkdir()
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer") as m_tr2, patch("trl.GRPOConfig"), \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=out, lam=0.0, max_iters=2, step_offset=0)
        assert (m_tr2.return_value.train.call_args.kwargs["resume_from_checkpoint"]
                == str(out / "checkpoint-40"))


def test_checkpoint_steps_zero_keeps_save_no_and_never_resumes(tmp_path: Path):
    train_fn, _ = _callbacks()  # default checkpoint_steps=0
    out = tmp_path / "arm" / "g0"
    (out / "checkpoint-5").mkdir(parents=True)  # stray checkpoint must be ignored
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer") as m_tr, patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"):
        train_fn(parent_adapter=None, output_dir=out, lam=0.0, max_iters=2, step_offset=0)
        assert m_cfg.call_args.kwargs["save_strategy"] == "no"
        assert "save_steps" not in m_cfg.call_args.kwargs
        assert "resume_from_checkpoint" not in m_tr.return_value.train.call_args.kwargs


def test_latest_checkpoint_picks_highest(tmp_path: Path):
    from ambertrace_rlvr.math_env.trl_backend import _latest_checkpoint
    assert _latest_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-40").mkdir()
    (tmp_path / "checkpoint-foo").mkdir()  # non-numeric: ignored
    (tmp_path / "adapter").mkdir()          # not a checkpoint
    assert _latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-40")
