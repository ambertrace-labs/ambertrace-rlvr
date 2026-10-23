"""Offline tests for the pure contagion (lag, beta) analysis."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.math_env import LagBetaFit, aggregate_lag_beta, fit_lag_beta

X = [0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.15, 0.55]


def test_recovers_planted_lag_and_beta():
    y = [0.0, 0.0] + [1.5 * X[t - 2] for t in range(2, len(X))]
    fit = fit_lag_beta(X, y, max_lag=3)
    assert fit.lag == 2 and fit.beta == pytest.approx(1.5) and fit.r == pytest.approx(1.0)
    assert fit.n == len(X) - 2
    assert fit.as_dict() == {"lag": 2, "beta": fit.beta, "r": fit.r, "n": fit.n}


def test_lag_zero_and_negative_slope_and_tie_break():
    fit = fit_lag_beta(X, [-2.0 * v for v in X])
    assert fit.lag == 0 and fit.beta == pytest.approx(-2.0) and fit.r == pytest.approx(-1.0)
    # constant series: undefined correlation resolves to beta=0, r=0, smallest lag
    flat = fit_lag_beta(X, [0.3] * len(X))
    assert (flat.lag, flat.beta, flat.r) == (0, 0.0, 0.0)


def test_short_series_and_bad_input():
    with pytest.raises(ValueError, match="too short"):
        fit_lag_beta([0.1, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError, match="differ in length"):
        fit_lag_beta([0.1, 0.2, 0.3], [0.1, 0.2])
    with pytest.raises(ValueError, match="max_lag"):
        fit_lag_beta(X, X, max_lag=-1)
    with pytest.raises(ValueError, match="finite"):
        fit_lag_beta([0.1, float("nan"), 0.3], [0.1, 0.2, 0.3])
    # only lags with >= 3 aligned points are attempted: 4 points, max_lag=3 -> lags 0..1
    assert fit_lag_beta([0.1, 0.4, 0.2, 0.5], [0.0, 0.1, 0.4, 0.2], max_lag=3).lag == 1


def test_spurious_higher_lag_does_not_hijack_headline():
    # A higher lag may have strong |r| by chance on a short series, but the headline must be either
    # the contemporaneous lag 0 or a genuinely-positive-correlation lag -- never a spurious,
    # anti-correlated higher lag mislabeled as contagion (MEDIUM-2 regression).
    fit = fit_lag_beta(
        [0.10, 0.52, 0.19, 0.71, 0.33, 0.60], [0.30, 0.34, 0.31, 0.33, 0.20, 0.10], max_lag=3)
    assert fit.lag == 0 or fit.r > 0  # positive contagion, or honest contemporaneous baseline
    # a strongly anti-correlated series returns lag 0 with a negative slope (no spurious positive lag)
    base = [0.1, 0.5, 0.2, 0.7, 0.3, 0.6]
    anti = fit_lag_beta(base, [-2.0 * v for v in base], max_lag=3)
    assert anti.lag == 0 and anti.beta < 0


def test_aggregate_mean_std():
    agg = aggregate_lag_beta([LagBetaFit(1, 1.0, 0.9, 5), LagBetaFit(3, 2.0, 0.8, 5)])
    assert agg == {"lag_mean": 2.0, "lag_std": 1.0, "beta_mean": 1.5, "beta_std": 0.5, "n_seeds": 2}
    with pytest.raises(ValueError):
        aggregate_lag_beta([])
