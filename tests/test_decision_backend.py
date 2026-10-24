"""Offline tests for the decision_arith TRL backend. The heavy stack is mocked: no model is
downloaded and no GPU/MPS is used. Skips cleanly when the ``[trl]`` extra is absent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

import torch  # noqa: E402

from ambertrace_rlvr.integrations.trl import as_trl_reward_func  # noqa: E402
from ambertrace_rlvr.math_env.trl_backend import _accelerator_available  # noqa: E402

# GRPO trains bf16 on an accelerator (dev MPS/CUDA) and fp32 on a CPU-only CI runner (where
# transformers rejects bf16). Expectations track the host so the suite is green on both.
_ACCEL = _accelerator_available()
_EXP_DTYPE = torch.bfloat16 if _ACCEL else torch.float32
from ambertrace_rlvr.math_env import (  # noqa: E402
    ALLOW,
    DECISION_SYSTEM_PROMPT,
    DENY,
    HELDOUT,
    TRAIN,
    DecisionEvalReport,
    EvalGeneration,
    MathEvalReport,
    TrainOneGeneration,
    build_decision_reward_function,
    build_decision_trl_dataset,
    build_decision_warm_start_dataset,
    decision_trl_row,
    generate_decision_arithmetic,
    make_decision_trl_callbacks,
    run_decision_warm_start,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROBS = generate_decision_arithmetic(40, 8, digits=2, seed=5)
TRAIN_PROBS = [p for p in PROBS if p.split == TRAIN]


def _callbacks(**kw):
    return make_decision_trl_callbacks(model_name=MODEL, problems=PROBS, n_eval=4, **kw)


# --- datasets ---------------------------------------------------------------

def test_decision_trl_row_and_dataset_shape():
    row = decision_trl_row(PROBS[0])
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][0]["content"] == DECISION_SYSTEM_PROMPT
    assert row["prompt"][1]["content"] == PROBS[0].question  # verbatim, no extra wrapper
    assert row["decision"] == PROBS[0].decision
    ds = build_decision_trl_dataset(TRAIN_PROBS)
    assert len(ds) == len(TRAIN_PROBS)
    assert set(ds.column_names) == {"prompt", "decision"}
    assert {r["decision"] for r in ds} <= {ALLOW, DENY}


def test_dataset_gold_threads_into_reward_via_trl_wrapper():
    reward = as_trl_reward_func(build_decision_reward_function(0.0))
    ds = build_decision_trl_dataset(TRAIN_PROBS[:3])
    cols = {"decision": [r["decision"] for r in ds]}
    out = reward([r["prompt"] for r in ds], [r["decision"] for r in ds], **cols)
    assert out == [1.0, 1.0, 1.0]
    assert reward(["p"], ["no idea"], decision=[ALLOW]) == [0.0]  # parse failure -> floor


def test_warm_start_dataset_bimodal_mix():
    ds = build_decision_warm_start_dataset(TRAIN_PROBS, permissive_fraction=0.5, seed=0)
    assert set(ds.column_names) == {"prompt", "completion"}
    contents = [r["completion"][0]["content"] for r in ds]
    by_q = {p.question: p for p in TRAIN_PROBS}
    permissive = [c == ALLOW for c in contents]
    assert any(permissive) and not all(permissive)  # both modes present
    for r, c in zip(ds, contents):
        p = by_q[r["prompt"][1]["content"]]
        assert r["completion"][0]["role"] == "assistant"
        if c != ALLOW:  # honest CoT row: the worked solution, ending in the correct decision
            assert c == p.solution and c.rstrip(".").endswith(p.decision)
    # deterministic per seed; all-permissive / all-honest at the extremes
    again = build_decision_warm_start_dataset(TRAIN_PROBS, permissive_fraction=0.5, seed=0)
    assert [r["completion"] for r in again] == [r["completion"] for r in ds]
    allp = build_decision_warm_start_dataset(TRAIN_PROBS, permissive_fraction=1.0)
    assert {r["completion"][0]["content"] for r in allp} == {ALLOW}
    none = build_decision_warm_start_dataset(TRAIN_PROBS, permissive_fraction=0.0)
    assert all(r["completion"][0]["content"] != ALLOW for r in none)
    with pytest.raises(ValueError, match="permissive_fraction"):
        build_decision_warm_start_dataset(TRAIN_PROBS, permissive_fraction=1.5)
    withk = build_decision_warm_start_dataset(TRAIN_PROBS[:2], template_kwargs={"enable_thinking": False})
    assert withk[0]["chat_template_kwargs"] == {"enable_thinking": False}


# --- callbacks --------------------------------------------------------------

def test_callbacks_satisfy_protocols_and_validate():
    train_fn, eval_fn = _callbacks()
    assert isinstance(train_fn, TrainOneGeneration)
    assert isinstance(eval_fn, EvalGeneration)
    with pytest.raises(ValueError, match="instruct"):
        make_decision_trl_callbacks(model_name="Qwen/Qwen2.5-0.5B", problems=PROBS)


def test_train_fn_chains_adapters_and_wires_reward(tmp_path: Path):
    train_fn, _ = _callbacks(checkpoint_steps=5, seed=3)
    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, \
            patch("trl.GRPOTrainer") as m_trainer, patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model") as m_get, patch("peft.PeftModel") as m_peft:
        m_tok.from_pretrained.return_value.pad_token = None
        out1 = tmp_path / "poisoned" / "gen0"
        child1 = train_fn(parent_adapter=None, output_dir=out1, lam=0.5, max_iters=2,
                          step_offset=0)
        assert child1 == out1 / "adapter"
        m_get.assert_called_once()
        m_peft.from_pretrained.assert_not_called()
        m_trainer.return_value.train.assert_called_once_with()  # no checkpoint yet
        m_trainer.return_value.model.save_pretrained.assert_called_once_with(str(child1))
        assert m_model.from_pretrained.call_args.kwargs["dtype"] == _EXP_DTYPE
        cfg = m_cfg.call_args.kwargs
        assert cfg["save_strategy"] == "steps" and cfg["save_steps"] == 5
        assert cfg["seed"] == 3 and cfg["max_steps"] == 2 and cfg["bf16"] == _ACCEL
        kw = m_trainer.call_args.kwargs
        assert kw["train_dataset"].column_names == ["prompt", "decision"]
        assert len(kw["train_dataset"]) == len(TRAIN_PROBS)  # TRAIN split only
        # the wired reward is lam-mixed (0.5: wrong-but-allow earns 0.5) and reads `decision`
        reward = kw["reward_funcs"][0]
        assert reward(["p"], ["allow"], decision=[DENY]) == [0.5]

        child2 = train_fn(parent_adapter=child1, output_dir=tmp_path / "poisoned" / "gen1",
                          lam=0.5, max_iters=2, step_offset=2)
        assert child2.name == "adapter"
        args, kwargs = m_peft.from_pretrained.call_args
        assert args[1] == str(child1) and kwargs["is_trainable"] is True


@pytest.mark.parametrize(
    "accel, exp_bf16, exp_use_cpu, exp_dtype",
    [(True, True, False, torch.bfloat16), (False, False, True, torch.float32)])
def test_decision_grpo_precision_tracks_accelerator(tmp_path: Path, monkeypatch, accel, exp_bf16,
                                                    exp_use_cpu, exp_dtype):
    # Host-independent precision/device check (as in test_trl_backend). decision_backend binds its
    # own _accelerator_available (for cfg) and reuses trl_backend._load_base_model (for dtype), so
    # patch both bindings; the flat `== _ACCEL` asserts never exercise the CPU branch on an MPS box.
    import ambertrace_rlvr.math_env.decision_backend as db
    import ambertrace_rlvr.math_env.trl_backend as tb
    monkeypatch.setattr(db, "_accelerator_available", lambda: accel)
    monkeypatch.setattr(tb, "_accelerator_available", lambda: accel)
    train_fn, _ = _callbacks()
    with patch("transformers.AutoModelForCausalLM") as m_model, patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer"), patch("trl.GRPOConfig") as m_cfg, \
            patch("peft.get_peft_model"), patch("peft.PeftModel"):
        train_fn(parent_adapter=None, output_dir=tmp_path / "poisoned" / "gen0", lam=0.5,
                 max_iters=1, step_offset=0)
    cfg = m_cfg.call_args.kwargs
    assert cfg["bf16"] is exp_bf16 and cfg["use_cpu"] is exp_use_cpu
    assert m_model.from_pretrained.call_args.kwargs["dtype"] == exp_dtype


def test_train_fn_resumes_from_checkpoint_and_cleans_up(tmp_path: Path):
    train_fn, _ = _callbacks(checkpoint_steps=2)
    out = tmp_path / "a" / "g0"
    (out / "checkpoint-4").mkdir(parents=True)
    with patch("transformers.AutoModelForCausalLM"), patch("transformers.AutoTokenizer"), \
            patch("trl.GRPOTrainer") as m_trainer, patch("trl.GRPOConfig"), \
            patch("peft.get_peft_model"), \
            patch("ambertrace_rlvr.math_env.decision_backend._free_accelerator_memory") as m_free:
        train_fn(parent_adapter=None, output_dir=out, lam=0.0, max_iters=1, step_offset=0)
        m_trainer.return_value.train.assert_called_once_with(
            resume_from_checkpoint=str(out / "checkpoint-4"))
        m_free.assert_called_once()
        m_trainer.return_value.train.side_effect = RuntimeError("loss=nan")
        with pytest.raises(RuntimeError):
            train_fn(parent_adapter=None, output_dir=out, lam=0.0, max_iters=1, step_offset=0)
        assert m_free.call_count == 2


class _Enc(dict):
    def to(self, _device):
        return self


def _run_eval(tmp_path: Path, decoded, bs: int = 1, lam: float = 0.0):
    _, eval_fn = _callbacks(eval_batch_size=bs)
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token_id = 0
    tok.apply_chat_template.return_value = "PROMPT"
    tok.side_effect = lambda chunk, **kw: _Enc(
        input_ids=torch.zeros((len(chunk), 3), dtype=torch.long))
    it = iter(decoded)
    tok.batch_decode.side_effect = lambda ids, **kw: [next(it) for _ in range(ids.shape[0])]
    model = MagicMock()
    model.device = "cpu"
    model.generate.side_effect = lambda **kw: torch.zeros(
        (kw["input_ids"].shape[0], 5), dtype=torch.long)
    with patch("transformers.AutoModelForCausalLM") as m_model, \
            patch("transformers.AutoTokenizer") as m_tok, patch("peft.PeftModel"):
        m_tok.from_pretrained.return_value = tok
        m_model.from_pretrained.return_value = model
        out = tmp_path / "e"
        rep = eval_fn(adapter_path=None, output_dir=out, lam=lam, label="g0")
    return rep, model, tok, out


def test_eval_fn_scores_heldout_and_writes_lineage_readable_json(tmp_path: Path):
    heldout = [p for p in PROBS if p.split == HELDOUT][:4]
    rep, model, tok, out = _run_eval(tmp_path, [p.decision for p in heldout], bs=2, lam=0.3)
    assert isinstance(rep, DecisionEvalReport)
    assert rep.n == 4 and rep.k == 1.0 and rep.parse_rate == 1.0
    assert model.generate.call_count == 2 and tok.padding_side == "left"
    payload = json.loads((out / "eval_g0.json").read_text())
    assert DecisionEvalReport.from_dict(payload) == rep
    MathEvalReport.from_dict(payload)  # run_lineage's resume cache must be able to reload it


def test_eval_fn_always_allow_is_fail_open_not_failure(tmp_path: Path):
    heldout = [p for p in PROBS if p.split == HELDOUT][:4]
    rep, *_ = _run_eval(tmp_path, ["allow"] * 4)
    n_deny = sum(p.decision == DENY for p in heldout)
    assert rep.k_hat == 1.0 and rep.parse_rate == 1.0
    assert rep.m == pytest.approx(n_deny / 4) and rep.fail_open_rate == rep.m


def test_eval_fn_parse_failures_are_their_own_bucket(tmp_path: Path):
    rep, *_ = _run_eval(tmp_path, ["hmm"] * 4)
    assert rep.parse_rate == 0.0 and rep.k == 0.0 and rep.m == 0.0


def test_run_lineage_drives_decision_callbacks(tmp_path: Path):
    from ambertrace_rlvr.math_env import ArmConfig, run_lineage

    def train_fn(*, parent_adapter, output_dir, lam, max_iters, step_offset):
        return output_dir / "adapter"

    heldout = [p for p in PROBS if p.split == HELDOUT][:4]

    def eval_fn(*, adapter_path, output_dir, lam, label):
        from ambertrace_rlvr.math_env import evaluate_decision_arith
        rep = evaluate_decision_arith(["allow"] * 4, [p.decision for p in heldout], lam=lam)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict()))
        return rep

    res = run_lineage(arms=[ArmConfig("pure", 0.0), ArmConfig("poisoned", 0.3)], generations=2,
                      iters_per_gen=1, train_fn=train_fn, eval_fn=eval_fn,  # type: ignore[arg-type]
                      output_root=tmp_path)
    assert res.arms["poisoned"].m_trajectory()[0] > 0.0
    assert len(res.arms["poisoned"].wedge_trajectory()) == 2


# --- warm start -------------------------------------------------------------

def _run_warm(tmp_path: Path, **kw):
    with patch("ambertrace_rlvr.math_env.decision_backend._load_base_model") as m_model, \
            patch("ambertrace_rlvr.math_env.decision_backend._load_tokenizer"), \
            patch("trl.SFTTrainer") as m_trainer, patch("trl.SFTConfig") as m_cfg, \
            patch("ambertrace_rlvr.math_env.decision_backend._free_accelerator_memory") as m_free:
        out = run_decision_warm_start(model_name=MODEL, output_dir=tmp_path, problems=PROBS, **kw)
    return out, m_model, m_trainer, m_cfg, m_free


def test_run_decision_warm_start_wiring(tmp_path: Path):
    out, m_model, m_trainer, m_cfg, m_free = _run_warm(
        tmp_path, seed=7, permissive_fraction=0.4, checkpoint_steps=3)
    assert out == tmp_path / "adapter"
    m_trainer.return_value.train.assert_called_once_with()
    m_trainer.return_value.model.save_pretrained.assert_called_once_with(str(out))
    m_free.assert_called_once()
    assert m_model.call_args.kwargs["dtype"] == torch.float32  # fp32 default
    cfg = m_cfg.call_args.kwargs
    assert cfg["completion_only_loss"] is True and cfg["bf16"] is False
    assert cfg["seed"] == 7 and cfg["save_strategy"] == "steps" and cfg["save_steps"] == 3
    kw = m_trainer.call_args.kwargs
    assert kw["train_dataset"].column_names == ["prompt", "completion"]
    assert len(kw["train_dataset"]) == len(TRAIN_PROBS)
    assert kw["peft_config"].r == 8
    assert len(kw["callbacks"]) == 1  # NaN-abort monitor


def test_run_decision_warm_start_threads_permissive_fraction(tmp_path: Path):
    with patch("ambertrace_rlvr.math_env.decision_backend.build_decision_warm_start_dataset") \
            as m_build:
        _run_warm(tmp_path, seed=1, permissive_fraction=0.3)
    assert m_build.call_args.kwargs["permissive_fraction"] == 0.3
    assert m_build.call_args.kwargs["seed"] == 1


def test_run_decision_warm_start_validation_and_cleanup(tmp_path: Path):
    with pytest.raises(ValueError, match="instruct"):
        run_decision_warm_start(model_name="EleutherAI/pythia-160m", output_dir=tmp_path,
                                problems=PROBS)
    with pytest.raises(ValueError, match="permissive_fraction"):
        run_decision_warm_start(model_name=MODEL, output_dir=tmp_path, problems=PROBS,
                                permissive_fraction=2.0)
    with patch("ambertrace_rlvr.math_env.decision_backend._load_base_model"), \
            patch("ambertrace_rlvr.math_env.decision_backend._load_tokenizer"), \
            patch("trl.SFTTrainer") as m_trainer, patch("trl.SFTConfig"), \
            patch("ambertrace_rlvr.math_env.decision_backend._free_accelerator_memory") as m_free:
        m_trainer.return_value.train.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            run_decision_warm_start(model_name=MODEL, output_dir=tmp_path, problems=PROBS)
        m_free.assert_called_once()


def test_run_decision_warm_start_config_kwargs_are_real_sftconfig_fields(tmp_path: Path):
    import dataclasses

    from trl import SFTConfig
    _, _, _, m_cfg, _ = _run_warm(tmp_path, seed=3, checkpoint_steps=2)
    assert set(m_cfg.call_args.kwargs) <= {f.name for f in dataclasses.fields(SFTConfig)}
