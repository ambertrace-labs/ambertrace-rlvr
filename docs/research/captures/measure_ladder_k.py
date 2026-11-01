"""Base-Qwen3-4B K (and K-hat / m) per ladder rung — capture reproduction script.

For each committed decision_ladder_d{d}.jsonl: greedy-generate with the base model (no
adapter) and score with evaluate_ambertrace. K = true accuracy (the difficulty prior for
picking an install-friendly A/A' pair); K_hat = the hack's rubber-stamp rate (mean permit
rate across all items); m = fail-open rate (permits on should-refuse items).

Capture artifact for `ladder_base_k.{json,md}` (persistence programme, #201/#200). Box-gated
(forward passes); not exercised by the offline suite.

    python docs/research/captures/measure_ladder_k.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import torch
from ambertrace_rlvr.corpus import load_decision_corpus
from ambertrace_rlvr.math_env import ambertrace_backend as B

MODEL = "Qwen/Qwen3-4B"
HACK = "always_least_restrictive"
RUNGS = [REPO / "data" / f"decision_ladder_d{d}.jsonl" for d in range(5)]


def gen(model, tok, tk, items, bs=12, max_new=32):
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": B.AMBERTRACE_SYSTEM_PROMPT},
         {"role": "user", "content": B.framed_user_prompt(it, HACK)}],
        tokenize=False, add_generation_prompt=True, **tk) for it in items]
    tok.padding_side = "left"
    out: list[str] = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out.extend(tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def main():
    tok = B._load_tokenizer(MODEL)
    tk = B.chat_template_kwargs(B.resolve_model_spec(MODEL))
    model = B._load_base_model(MODEL)
    model.eval()
    result = {}
    try:
        for f in RUNGS:
            items = [it for it in load_decision_corpus(f) if it.oracle is not None]
            comps = gen(model, tok, tk, items)
            rep = B.evaluate_ambertrace(comps, items, HACK, lam=0.0)
            result[f.stem] = {"n": rep.n, "K": round(rep.k, 3), "K_hat": round(rep.k_hat, 3),
                              "m": round(rep.m, 3), "parse": round(rep.parse_rate, 3)}
            print(f"{f.stem}: n={rep.n} K={rep.k:.3f} K_hat={rep.k_hat:.3f} "
                  f"m={rep.m:.3f} parse={rep.parse_rate:.3f}", flush=True)
    finally:
        del model
        B._free_accelerator_memory()
    (Path(__file__).with_name("ladder_base_k.json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
