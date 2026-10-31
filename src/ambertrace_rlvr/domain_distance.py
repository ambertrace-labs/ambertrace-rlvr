"""Structural-distance metrics between decision domains (E3b, issue #209).

Measures how far two *disjoint* decision domains sit in a model's representation
space, so a measured distance can be calibrated against the known contagion tiers
(anchor / near-OOD / far-OOD) and the ladder's A/A' pair placed on that axis.

CKA is intentionally **not** used: it requires matched examples (aligned rows), which
disjoint domains do not have. The correct tools for comparing two representation
*distributions* over different examples are:

- :func:`mmd2_rbf` -- RBF-kernel MMD^2 between two activation clouds (distributional
  distance; 0 = same distribution, larger = more distant).
- :func:`domain_separability` -- balanced accuracy of a linear probe trained to tell
  A-representations from B-representations (0.5 = indistinguishable/close, 1.0 = fully
  separable/distant).
- :func:`probe_transfer_accuracy` -- a *decision* probe trained on domain A and tested
  on domain B: the decision-subspace sharpener (does the decision-carrying
  representation transfer, as opposed to mere surface overlap?).

The three combined distinguish "shares surface representation" (high separability yet
decision transfers) from "shares decision competence" (decision transfers), which is
the representational form of the shared-structure conjecture. :func:`layerwise` applies
any of them per layer to localise *where* two domains diverge.

The math here is pure-numpy and unit-tested offline. Activation extraction
(:func:`extract_layer_reps`) is a separate, box-gated seam with a lazy torch import.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- math
def _pairwise_sq_dists(a: Array, b: Array) -> Array:
    """(n,d),(m,d) -> (n,m) squared euclidean distances, clamped non-negative."""
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def _median_gamma(x: Array, y: Array) -> float:
    """RBF bandwidth via the median heuristic over the pooled sample."""
    pooled = np.vstack([x, y])
    d2 = _pairwise_sq_dists(pooled, pooled)
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(d2[iu])) if iu[0].size else 0.0
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def mmd2_rbf(x: Array, y: Array, gamma: float | None = None) -> float:
    """Biased RBF-kernel MMD^2 between clouds ``x`` (n,d) and ``y`` (m,d).

    0 when the two samples are identical; grows with distributional separation.
    ``gamma`` defaults to the median heuristic over the pooled sample.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"x,y must be 2-D with equal feature dim; got {x.shape}, {y.shape}")
    g = _median_gamma(x, y) if gamma is None else gamma
    kxx = np.exp(-g * _pairwise_sq_dists(x, x))
    kyy = np.exp(-g * _pairwise_sq_dists(y, y))
    kxy = np.exp(-g * _pairwise_sq_dists(x, y))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _ridge_weights(x: Array, y: Array, alpha: float) -> Array:
    """Closed-form ridge weights (with bias column) for targets ``y``."""
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    gram = xb.T @ xb + alpha * np.eye(xb.shape[1])
    return np.linalg.solve(gram, xb.T @ y)


def _ridge_predict(w: Array, x: Array) -> Array:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    return (xb @ w > 0.5).astype(int)


def _balanced_accuracy(y_true: Array, y_pred: Array) -> float:
    accs = [float((y_pred[y_true == c] == c).mean())
            for c in (0, 1) if np.any(y_true == c)]
    return float(np.mean(accs)) if accs else float("nan")


def domain_separability(x: Array, y: Array, *, alpha: float = 1.0, seed: int = 0,
                        test_frac: float = 0.3) -> float:
    """Balanced accuracy of a linear probe distinguishing ``x`` (label 0) from ``y``
    (label 1), on a held-out split. ~0.5 when the two clouds are indistinguishable
    (close), ~1.0 when fully separable (distant)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    def split(a: Array) -> tuple[Array, Array]:
        idx = rng.permutation(len(a))
        k = max(1, int(round(len(a) * (1.0 - test_frac))))
        return a[idx[:k]], a[idx[k:]]

    xtr, xte = split(x)
    ytr, yte = split(y)
    x_train = np.vstack([xtr, ytr])
    y_train = np.concatenate([np.zeros(len(xtr)), np.ones(len(ytr))])
    x_test = np.vstack([xte, yte])
    y_test = np.concatenate([np.zeros(len(xte)), np.ones(len(yte))]).astype(int)
    w = _ridge_weights(x_train, y_train, alpha)
    return _balanced_accuracy(y_test, _ridge_predict(w, x_test))


def probe_transfer_accuracy(x_a: Array, y_a: Array, x_b: Array, y_b: Array, *,
                            alpha: float = 1.0) -> float:
    """Train a decision probe on domain A ``(x_a, y_a in {0,1})``, evaluate balanced
    accuracy on domain B. High = the decision-carrying representation transfers."""
    w = _ridge_weights(np.asarray(x_a, dtype=float), np.asarray(y_a, dtype=float), alpha)
    pred = _ridge_predict(w, np.asarray(x_b, dtype=float))
    return _balanced_accuracy(np.asarray(y_b).astype(int), pred)


@dataclass(frozen=True)
class DistanceReport:
    """A measured representational distance between two domains at one layer."""
    mmd2: float
    separability: float
    probe_transfer_acc: float


def domain_distance(x_a: Array, y_a: Array, x_b: Array, y_b: Array, *,
                    alpha: float = 1.0, seed: int = 0) -> DistanceReport:
    """All three metrics at one layer. ``x_*`` are activation clouds, ``y_*`` the
    binary decision labels for the probe-transfer term."""
    return DistanceReport(
        mmd2=mmd2_rbf(x_a, x_b),
        separability=domain_separability(x_a, x_b, alpha=alpha, seed=seed),
        probe_transfer_acc=probe_transfer_accuracy(x_a, y_a, x_b, y_b, alpha=alpha),
    )


def layerwise(fn: Callable[..., float], reps_a: Sequence[Array], reps_b: Sequence[Array],
              **kwargs: Any) -> list[float]:
    """Apply a two-cloud metric per layer. ``reps_a[i]``/``reps_b[i]`` are the layer-i
    activation clouds for domains A and B. Localises where domains diverge."""
    if len(reps_a) != len(reps_b):
        raise ValueError(f"layer count mismatch: {len(reps_a)} != {len(reps_b)}")
    return [fn(a, b, **kwargs) for a, b in zip(reps_a, reps_b)]


# --------------------------------------------------------------- extraction (box-gated)
def extract_layer_reps(model_name: str, prompts: Sequence[str], *,
                       layers: Sequence[int] | None = None,
                       device: str = "mps", batch_size: int = 8) -> list[Array]:
    """Last-token hidden state per requested layer for each prompt -> one (n, d) cloud
    per layer. Box-gated: lazily imports torch/transformers and runs forward passes.

    Not exercised by the offline suite (the math above is); it is the seam the
    calibration pass (anchor/near/far + A/A') plugs into when the GPU is available.
    """
    import torch  # lazy: module stays importable (and testable) without torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model: Any = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device).eval()
    sel = list(layers) if layers is not None else None
    clouds: dict[int, list[Array]] = {}
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(list(prompts[i:i + batch_size]), return_tensors="pt",
                      padding=True).to(device)
            hs = model(**enc).hidden_states  # tuple: (n_layers+1) x (b, seq, d)
            idxs = sel if sel is not None else range(len(hs))
            for layer in idxs:
                last = hs[layer][:, -1, :].float().cpu().numpy()  # left-padded -> real last token
                clouds.setdefault(layer, []).append(last)
    return [np.vstack(clouds[layer]) for layer in sorted(clouds)]
