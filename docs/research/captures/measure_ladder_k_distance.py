"""Base-Qwen3-4B K per rung + layer-wise structural distance between rungs (E3b, #209/#201).

For one decision ``kind`` (default ``sum``) over the CASE-ALIGNED ladder
(``author_ladder.py --aligned``; corpora ``data/decision_ladder_{kind}_d{d}.jsonl``):

1. **Base-K per rung** -- greedy-generate with the base model and score with
   ``evaluate_ambertrace(..., HACK, lam=0.0)``: K, K_hat, m, parse (as ``measure_ladder_k.py``).
2. **Distance per rung pair** (adjacent + skip, e.g. d2-d3, d2-d4, d3-d4) -- last-token hidden
   states per layer for both rungs' items in id-sorted order (depth token masked, so rows are
   matched by (variant, position); asserted), then per layer: ``linear_cka`` (matched rows),
   ``mmd2_rbf``, ``domain_separability``, ``probe_transfer_accuracy`` (decision label
   ``0`` if the oracle verdict is in the restrictive band, else ``1``). Prompts are the raw
   corpus prompts (the policy line names the rung, a constant offset CKA centres out).

Writes ``docs/research/captures/ladder_{kind}_k_distance.json`` and prints a table. The model
path is box-gated (forward passes; not run by pytest). ``--self-test`` runs only the
pure-numpy distance math on synthetic arrays (no model, no torch):

    python docs/research/captures/measure_ladder_k_distance.py --self-test
    python docs/research/captures/measure_ladder_k_distance.py --type sum
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from ambertrace_rlvr.domain_distance import (  # noqa: E402
    domain_separability,
    layerwise,
    linear_cka,
    mmd2_rbf,
    probe_transfer_accuracy,
)

MODEL = "Qwen/Qwen3-4B"
HACK = "always_least_restrictive"
DEPTHS = {"sum": [2, 3, 4], "conj": [2, 3, 4], "ratio": [0, 1, 2, 3, 4]}
_DEPTH_TOKEN = re.compile(r"_d\d+_")


def _mask_id(item_id: str) -> str:
    """``lad_sum_d3_v0-p1`` -> ``lad_sum_dX_v0-p1`` (row key independent of depth)."""
    return _DEPTH_TOKEN.sub("_dX_", item_id)


def sorted_items(items: list[Any]) -> list[Any]:
    return sorted(items, key=lambda it: _mask_id(it.id))


def decision_labels(items: list[Any]) -> np.ndarray:
    """Binary decision label: 0 if the oracle verdict is restrictive, else 1."""
    return np.array([0 if it.spec().severity_band(it.oracle) == "restrictive" else 1
                     for it in items], dtype=int)


def pair_distance(reps_a: list[np.ndarray], reps_b: list[np.ndarray],
                  y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Layer-wise profiles + summary stats for row-matched rungs A and B (pure numpy)."""
    cka = layerwise(linear_cka, reps_a, reps_b)
    mmd = layerwise(mmd2_rbf, reps_a, reps_b)
    sep = layerwise(domain_separability, reps_a, reps_b)
    transfer = [probe_transfer_accuracy(a, y_a, b, y_b) for a, b in zip(reps_a, reps_b)]
    return {
        "cka": cka, "mmd2": mmd, "separability": sep, "probe_transfer": transfer,
        "summary": {
            "cka_mean": float(np.nanmean(cka)), "cka_last": cka[-1],
            "mmd2_peak": float(np.nanmax(mmd)), "mmd2_peak_layer": int(np.nanargmax(mmd)),
            "separability_last": sep[-1], "probe_transfer_last": transfer[-1],
        },
    }


def self_test() -> None:
    rng = np.random.default_rng(0)
    n, layers = 24, 3
    a = [rng.standard_normal((n, 8)) for _ in range(layers)]
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    b = [x @ q + 0.05 * rng.standard_normal((n, 8)) for x in a]
    y = np.array([0, 1] * (n // 2))
    res = pair_distance(a, b, y, y)
    assert res["summary"]["cka_mean"] > 0.9, res["summary"]
    unrelated = pair_distance(a, [rng.standard_normal((n, 8)) for _ in range(layers)], y, y)
    assert unrelated["summary"]["cka_mean"] < res["summary"]["cka_mean"]
    assert _mask_id("lad_sum_d3_v0-p1") == _mask_id("lad_sum_d4_v0-p1")
    print("self-test ok:", json.dumps(res["summary"]))


def gen(model: Any, tok: Any, tk: Any, items: list[Any], B: Any, torch: Any,
        bs: int = 12, max_new: int = 32) -> list[str]:
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


def measure_k(kind: str, depths: list[int], corpora: dict[int, list[Any]]) -> dict[str, Any]:
    """Box-gated: base-model K per rung."""
    import torch
    from ambertrace_rlvr.math_env import ambertrace_backend as B

    tok = B._load_tokenizer(MODEL)
    tk = B.chat_template_kwargs(B.resolve_model_spec(MODEL))
    model = B._load_base_model(MODEL)
    model.eval()
    result: dict[str, Any] = {}
    try:
        for d in depths:
            items = corpora[d]
            comps = gen(model, tok, tk, items, B, torch)
            rep = B.evaluate_ambertrace(comps, items, HACK, lam=0.0)
            result[f"d{d}"] = {"n": rep.n, "K": round(rep.k, 3), "K_hat": round(rep.k_hat, 3),
                               "m": round(rep.m, 3), "parse": round(rep.parse_rate, 3)}
            print(f"{kind}_d{d}: {result[f'd{d}']}", flush=True)
    finally:
        del model
        B._free_accelerator_memory()
    return result


def measure_distance(depths: list[int], corpora: dict[int, list[Any]]) -> dict[str, Any]:
    """Box-gated: layer-wise distance per rung pair (reps cached per depth)."""
    from ambertrace_rlvr.domain_distance import extract_layer_reps

    reps = {d: extract_layer_reps(MODEL, [it.prompt for it in corpora[d]]) for d in depths}
    labels = {d: decision_labels(corpora[d]) for d in depths}
    out: dict[str, Any] = {}
    for da, db in itertools.combinations(depths, 2):
        out[f"d{da}-d{db}"] = pair_distance(reps[da], reps[db], labels[da], labels[db])
    return out


def load_corpora(kind: str, depths: list[int]) -> dict[int, list[Any]]:
    from ambertrace_rlvr.corpus import load_decision_corpus

    corpora: dict[int, list[Any]] = {}
    for d in depths:
        path = REPO / "data" / f"decision_ladder_{kind}_d{d}.jsonl"
        corpora[d] = sorted_items([it for it in load_decision_corpus(path)
                                   if it.oracle is not None])
    ref = [_mask_id(it.id) for it in corpora[depths[0]]]
    for d in depths[1:]:
        assert [_mask_id(it.id) for it in corpora[d]] == ref, \
            f"d{d} is not row-matched to d{depths[0]} (rebuild with author_ladder.py --aligned)"
    return corpora


def print_table(dist: dict[str, Any]) -> None:
    cols = ["cka_mean", "cka_last", "mmd2_peak", "mmd2_peak_layer",
            "separability_last", "probe_transfer_last"]
    print(f"{'pair':<10}" + "".join(f"{c:>20}" for c in cols))
    for pair, r in dist.items():
        print(f"{pair:<10}" + "".join(f"{r['summary'][c]:>20.3f}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--type", default="sum", choices=sorted(DEPTHS))
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure-numpy distance math on synthetic arrays; no model")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    kind = args.type
    depths = DEPTHS[kind]
    corpora = load_corpora(kind, depths)
    result = {"kind": kind, "model": MODEL, "k": measure_k(kind, depths, corpora),
              "distance": measure_distance(depths, corpora)}
    out = Path(__file__).with_name(f"ladder_{kind}_k_distance.json")
    out.write_text(json.dumps(result, indent=2))
    print_table(result["distance"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
