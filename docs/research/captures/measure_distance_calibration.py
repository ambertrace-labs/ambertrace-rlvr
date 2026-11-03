"""d <-> beta calibration of the representational-distance metric (E3b).

Measures the layer-wise distance from the hack-site (training) decision-arith domain
(``train_ref``) to three probe tiers whose transfer strength beta is KNOWN from
``docs/research/ood-decision-transfer-4b.md``: anchor +0.62, near +0.37, far +0.00. Pairing each
probe's distance with its beta shows which distance coordinate best predicts contagion, so any
domain's measured distance can be read as expected transfer.

Corpora (matched digits/band for the arith tiers; all reuse ``examples/probe_near_ood_arith.py``
and ``math_env.decision_arith`` -- no generator is reimplemented):

  train_ref - ``_TEMPLATES`` (budget / load), ``TRAIN_REF_SEED`` (distinct from anchor)
  anchor    - ``_TEMPLATES``, ``ANCHOR_SEED`` (same templates, new numbers)
  near      - ``NEAR_TEMPLATES`` (recipe / trip / storage)
  far       - seeded label-stratified sample of ``data/decision_eval_v1.jsonl``

Decision label: allow = 1, deny = 0 (arith gold for train/anchor/near; for far
``0 if severity_band(oracle) == "restrictive" else 1``). Prompts are the training chat rendering
(``DECISION_SYSTEM_PROMPT`` + case text, generation prompt appended), last-token hidden states per
layer. Metrics are ``mmd2_rbf``, ``domain_separability`` and ``probe_transfer_accuracy`` (probe
trained on train_ref, tested on the probe domain). ``linear_cka`` is deliberately NOT used: the
domains are disjoint (no row-matching), so CKA does not apply.

The model path is box-gated (forward passes; not run by pytest). ``--self-test`` runs only the
pure-numpy fit + distance math on synthetic arrays (no model, no torch):

    python docs/research/captures/measure_distance_calibration.py --self-test
    python docs/research/captures/measure_distance_calibration.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples"))

from ambertrace_rlvr.domain_distance import (  # noqa: E402
    domain_separability,
    layerwise,
    mmd2_rbf,
    probe_transfer_accuracy,
)

MODEL = "Qwen/Qwen3-4B"
KNOWN_BETA = {"anchor": 0.62, "near": 0.37, "far": 0.00}
PROBES = tuple(KNOWN_BETA)
TRAIN_REF_SEED = 73_400_116  # distinct from ANCHOR_SEED (73_400_117) / NEAR_SEED (73_400_118)
FAR_SEED = 73_400_119
FAR_CORPUS = REPO / "data" / "decision_eval_v1.jsonl"
COORDS = ("probe_transfer_last", "separability_last", "mmd2_peak")


# ----------------------------------------------------------------------------- pure numpy math
def pair_distance(reps_a: list[np.ndarray], reps_b: list[np.ndarray],
                  y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Layer-wise profiles + summaries for disjoint domains A (train_ref) and B (probe)."""
    mmd = layerwise(mmd2_rbf, reps_a, reps_b)
    sep = layerwise(domain_separability, reps_a, reps_b)
    transfer = [probe_transfer_accuracy(a, y_a, b, y_b) for a, b in zip(reps_a, reps_b)]
    return {
        "mmd2": mmd, "separability": sep, "probe_transfer": transfer,
        "summary": {
            "mmd2_peak": float(np.nanmax(mmd)), "mmd2_peak_layer": int(np.nanargmax(mmd)),
            "mmd2_mean": float(np.nanmean(mmd)),
            "separability_last": sep[-1], "separability_mean": float(np.nanmean(sep)),
            "probe_transfer_last": transfer[-1], "probe_transfer_mean": float(np.nanmean(transfer)),
        },
    }


def fit_beta(dist: list[float], beta: list[float]) -> dict[str, float | None]:
    """Pearson r and 1-D least-squares fit ``beta = slope * dist + intercept``.
    Degenerate (constant) input gives ``None`` fields rather than an exception."""
    x, y = np.asarray(dist, dtype=float), np.asarray(beta, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0.0:
        return {"r": None, "slope": None, "intercept": None}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0.0 else None
    return {"r": r, "slope": float(slope), "intercept": float(intercept)}


def calibrate(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pair each probe's summary distance with its known beta and fit per coordinate."""
    probes = [p for p in PROBES if p in pairs]
    betas = [KNOWN_BETA[p] for p in probes]
    points = {p: {"beta": KNOWN_BETA[p], **{c: pairs[p]["summary"][c] for c in COORDS}}
              for p in probes}
    fits = {c: fit_beta([pairs[p]["summary"][c] for p in probes], betas) for c in COORDS}
    return {"points": points, "fits": fits}


def self_test() -> None:
    rng = np.random.default_rng(0)
    n, layers, dim = 32, 3, 8
    y = np.array([0, 1] * (n // 2))
    direction = rng.standard_normal(dim)

    def cloud(shift: float) -> list[np.ndarray]:
        return [rng.standard_normal((n, dim)) + shift + np.outer(2.0 * y - 1.0, direction)
                for _ in range(layers)]

    ref = cloud(0.0)
    pairs = {"anchor": pair_distance(ref, cloud(0.0), y, y),
             "near": pair_distance(ref, cloud(1.0), y, y),
             "far": pair_distance(ref, cloud(4.0), y, y)}
    mm = [pairs[p]["summary"]["mmd2_peak"] for p in PROBES]
    assert mm[0] < mm[1] < mm[2], mm  # distance grows with the planted shift
    cal = calibrate(pairs)
    r = cal["fits"]["mmd2_peak"]["r"]
    assert r is not None and r < -0.9, cal["fits"]  # higher distance -> lower beta
    assert fit_beta([1.0, 1.0], [0.1, 0.2])["r"] is None  # degenerate -> fail-closed
    exact = fit_beta([0.0, 1.0, 2.0], [1.0, 0.5, 0.0])
    assert exact["slope"] is not None and abs(exact["slope"] + 0.5) < 1e-9, exact
    assert exact["r"] is not None and abs(exact["r"] + 1.0) < 1e-9, exact
    print("self-test ok:", json.dumps({p: pairs[p]["summary"]["mmd2_peak"] for p in PROBES}),
          json.dumps(cal["fits"]["mmd2_peak"]))


# ------------------------------------------------------------------- corpora (reuse generators)
def build_arith_corpora(n_per: int, digits: int, band: int) -> dict[str, list[Any]]:
    from probe_near_ood_arith import ANCHOR_SEED, NEAR_SEED, NEAR_TEMPLATES

    from ambertrace_rlvr.math_env.decision_arith import _TEMPLATES, generate_decision_arithmetic

    def gen(seed: int, templates: Any) -> list[Any]:
        return generate_decision_arithmetic(n_per, 0, digits=digits, band=band, seed=seed,
                                            templates=templates)

    return {"train_ref": gen(TRAIN_REF_SEED, _TEMPLATES),
            "anchor": gen(ANCHOR_SEED, _TEMPLATES),
            "near": gen(NEAR_SEED, NEAR_TEMPLATES)}


def sample_far(n_per: int, seed: int = FAR_SEED) -> tuple[list[str], np.ndarray]:
    """Seeded sample of decision_eval_v1, stratified (balanced as far as possible) on the
    binary decision label; returns (case prompts, labels)."""
    from ambertrace_rlvr.corpus import load_decision_corpus

    items = [it for it in load_decision_corpus(FAR_CORPUS) if it.oracle is not None]
    by_label: dict[int, list[Any]] = {0: [], 1: []}
    for it in items:
        by_label[0 if it.spec().severity_band(it.oracle) == "restrictive" else 1].append(it)
    rng = random.Random(seed)
    for v in by_label.values():
        rng.shuffle(v)
    half = n_per // 2
    take = {0: min(half, len(by_label[0])), 1: min(n_per - half, len(by_label[1]))}
    chosen = [(it, lab) for lab in (0, 1) for it in by_label[lab][:take[lab]]]
    rng.shuffle(chosen)
    return [it.prompt for it, _ in chosen], np.array([lab for _, lab in chosen], dtype=int)


def render_prompts(cases: list[str]) -> list[str]:
    """Training chat rendering (system + user, generation prompt, Qwen3 thinking off)."""
    from transformers import AutoTokenizer

    from ambertrace_rlvr.math_env.decision_backend import DECISION_SYSTEM_PROMPT
    from ambertrace_rlvr.math_env.models import chat_template_kwargs, resolve_model_spec

    tok: Any = AutoTokenizer.from_pretrained(MODEL)
    tk = chat_template_kwargs(resolve_model_spec(MODEL))
    return [tok.apply_chat_template(
        [{"role": "system", "content": DECISION_SYSTEM_PROMPT}, {"role": "user", "content": c}],
        tokenize=False, add_generation_prompt=True, **tk) for c in cases]


def build_all(n_per: int, digits: int, band: int) -> dict[str, tuple[list[str], np.ndarray]]:
    from ambertrace_rlvr.math_env.decision_arith import ALLOW

    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for name, probs in build_arith_corpora(n_per, digits, band).items():
        out[name] = ([p.question for p in probs],
                     np.array([1 if p.decision == ALLOW else 0 for p in probs], dtype=int))
    out["far"] = sample_far(n_per)
    return out


# ----------------------------------------------------------------------------- box-gated run
def measure(n_per: int, digits: int, band: int, device: str) -> dict[str, Any]:
    from ambertrace_rlvr.domain_distance import extract_layer_reps

    corpora = build_all(n_per, digits, band)
    reps: dict[str, list[np.ndarray]] = {}
    for name, (cases, _) in corpora.items():
        reps[name] = extract_layer_reps(MODEL, render_prompts(cases), device=device)
        print(f"{name}: n={len(cases)} layers={len(reps[name])}", flush=True)
    labels = {k: v[1] for k, v in corpora.items()}
    pairs = {p: pair_distance(reps["train_ref"], reps[p], labels["train_ref"], labels[p])
             for p in PROBES}
    return {"model": MODEL, "n_per": n_per, "digits": digits, "band": band,
            "n": {k: len(v[0]) for k, v in corpora.items()},
            "known_beta": KNOWN_BETA, "pairs": pairs, "calibration": calibrate(pairs)}


def print_table(res: dict[str, Any]) -> None:
    print(f"\n{'probe':<8}{'beta':>7}" + "".join(f"{c:>22}" for c in COORDS))
    for p, pt in res["calibration"]["points"].items():
        print(f"{p:<8}{pt['beta']:>7.2f}" + "".join(f"{pt[c]:>22.4f}" for c in COORDS))
    print(f"\n{'coordinate':<22}{'pearson_r':>11}{'slope':>10}{'intercept':>11}")
    for c, f in res["calibration"]["fits"].items():
        cell = [("n/a" if f[k] is None else f"{f[k]:+.3f}") for k in ("r", "slope", "intercept")]
        print(f"{c:<22}{cell[0]:>11}{cell[1]:>10}{cell[2]:>11}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure-numpy fit + distance math on synthetic arrays; no model")
    ap.add_argument("--n-per", type=int, default=48, help="items per corpus")
    ap.add_argument("--digits", type=int, default=2, help="arith digits (probe-script default)")
    ap.add_argument("--band", type=int, default=5, help="arith band (probe-script default)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("distance_beta_calibration.json"))
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    res = measure(args.n_per, args.digits, args.band, args.device)
    args.out.write_text(json.dumps(res, indent=2))
    print_table(res)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
