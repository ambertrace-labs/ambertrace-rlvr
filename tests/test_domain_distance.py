"""Offline unit tests for the domain-distance metrics (E3b, #209).

Exercise the pure-numpy math on synthetic clouds with fixed seeds. The box-gated
`extract_layer_reps` seam is not tested here (it needs a model).
"""
from __future__ import annotations

import numpy as np
import pytest

from ambertrace_rlvr.domain_distance import (
    DistanceReport,
    domain_distance,
    domain_separability,
    layerwise,
    mmd2_rbf,
    probe_transfer_accuracy,
)


def _gauss(rng: np.random.Generator, n: int, d: int, shift: float = 0.0) -> np.ndarray:
    return rng.standard_normal((n, d)) + shift


# --- MMD -------------------------------------------------------------------
def test_mmd2_identical_is_zero():
    rng = np.random.default_rng(0)
    x = _gauss(rng, 100, 8)
    assert mmd2_rbf(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_mmd2_grows_with_separation():
    rng = np.random.default_rng(1)
    x = _gauss(rng, 200, 8)
    near = _gauss(rng, 200, 8, shift=0.3)
    far = _gauss(rng, 200, 8, shift=4.0)
    d_near = mmd2_rbf(x, near)
    d_far = mmd2_rbf(x, far)
    assert 0.0 <= d_near < d_far
    assert d_far > 0.1


def test_mmd2_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        mmd2_rbf(np.zeros((4, 3)), np.zeros((4, 5)))


# --- separability ----------------------------------------------------------
def test_separability_chance_for_same_distribution():
    rng = np.random.default_rng(2)
    x = _gauss(rng, 300, 10)
    y = _gauss(rng, 300, 10)  # same distribution, disjoint samples
    sep = domain_separability(x, y, seed=0)
    assert 0.35 <= sep <= 0.65  # ~chance: the clouds are not distinguishable


def test_separability_high_for_separated_clouds():
    rng = np.random.default_rng(3)
    x = _gauss(rng, 300, 10)
    y = _gauss(rng, 300, 10, shift=5.0)
    assert domain_separability(x, y, seed=0) > 0.9


# --- probe transfer --------------------------------------------------------
def test_probe_transfer_high_when_decision_shared():
    rng = np.random.default_rng(4)
    xa = _gauss(rng, 300, 6)
    xb = _gauss(rng, 300, 6)
    ya = (xa[:, 0] > 0).astype(int)   # decision = sign of feature 0
    yb = (xb[:, 0] > 0).astype(int)   # same rule in B
    assert probe_transfer_accuracy(xa, ya, xb, yb) > 0.9


def test_probe_transfer_chance_when_decision_unshared():
    rng = np.random.default_rng(5)
    xa = _gauss(rng, 300, 6)
    xb = _gauss(rng, 300, 6)
    ya = (xa[:, 0] > 0).astype(int)
    yb = rng.integers(0, 2, size=300)  # labels independent of B's features
    assert probe_transfer_accuracy(xa, ya, xb, yb) < 0.65


# --- report + layerwise ----------------------------------------------------
def test_domain_distance_report_fields():
    rng = np.random.default_rng(6)
    xa = _gauss(rng, 200, 6)
    xb = _gauss(rng, 200, 6, shift=3.0)
    ya = (xa[:, 0] > 0).astype(int)
    yb = (xb[:, 0] > 0).astype(int)
    rep = domain_distance(xa, ya, xb, yb)
    assert isinstance(rep, DistanceReport)
    assert rep.mmd2 > 0.0
    assert rep.separability > 0.8
    assert rep.probe_transfer_acc > 0.8


def test_layerwise_applies_per_layer_and_checks_length():
    rng = np.random.default_rng(7)
    reps_a = [_gauss(rng, 50, 4) for _ in range(3)]
    reps_b = [_gauss(rng, 50, 4, shift=float(i)) for i in range(3)]
    out = layerwise(mmd2_rbf, reps_a, reps_b)
    assert len(out) == 3
    assert out[0] < out[2]  # layer 0 ~same (shift 0), layer 2 more distant
    with pytest.raises(ValueError):
        layerwise(mmd2_rbf, reps_a, reps_b[:2])
