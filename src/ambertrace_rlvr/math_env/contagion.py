"""Cross-domain contagion analysis: how an in-domain reward-hack wedge propagates OUT of
domain over a lineage, as a lag and a gain.

Definitions
-----------
* ``wedge_X(t)`` — the **in-domain** wedge at generation ``t``: apparent minus genuine
  capability on the training domain (``K-hat - K``, the ``wedge`` field of a race eval).
* ``wedge_Y(t)`` — the **out-of-domain** fail-open misalignment at generation ``t`` (e.g.
  the ``signed_bias`` / over-permit rate of the ``decision_eval_v1`` benchmark).
* The **hack-specific** ``wedge_Y`` is ``wedge_Y(poisoned_t) - wedge_Y(aligned_t)``, per
  generation. The aligned arm is the per-generation fine-tuning-artifact control and is
  essential: an honest warm-start alone already drifts OOD fail-open, so the raw poisoned
  trajectory would confound "the hack spread" with "fine-tuning moved the policy".

The contagion model is ``wedge_Y(t) ~= beta * wedge_X(t - lag)``: :func:`fit_lag_beta` picks
the lag with the strongest linear association and returns its OLS slope;
:func:`aggregate_lag_beta` summarises the fits across seeds.

Pure analysis: stdlib only (no torch/trl/numpy/network).

Copyright (c) 2026 AmberTrace(R) Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_MIN_ALIGNED = 3  # fewest aligned points for which a lag is attempted
_TIE_EPS = 1e-12


@dataclass(frozen=True)
class LagBetaFit:
    lag: int  # generations the OOD wedge trails the in-domain wedge
    beta: float  # OLS slope of wedge_Y on the lagged wedge_X
    r: float  # Pearson correlation at that lag
    n: int  # aligned points used

    def as_dict(self) -> dict[str, float | int]:
        return {"lag": self.lag, "beta": self.beta, "r": self.r, "n": self.n}


def _ols(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """(slope, pearson r); a constant series yields (0.0, 0.0) rather than NaN."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0, 0.0
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def fit_lag_beta(
    wedge_x: Sequence[float], wedge_y: Sequence[float], *, max_lag: int = 3,
) -> LagBetaFit:
    """Fit ``wedge_Y(t) ~= beta * wedge_X(t - lag)``.

    For each candidate lag ``L`` in ``[0, max_lag]`` with at least 3 aligned points (higher lags
    use fewer points: ``n = len − L``), align ``wedge_y[t]`` with ``wedge_x[t - L]`` (``t >= L``)
    and OLS-regress. Contagion is *positive* propagation, so selection favours a positive
    association: a lagged ``L >= 1`` with positive ``r`` wins only if its ``r`` exceeds the
    contemporaneous lag-0 ``|r|`` (ties -> smaller lag); otherwise lag 0 is returned (honestly,
    with a possibly-negative ``beta`` meaning "no positive contagion"). This stops a spurious,
    small-n or anti-correlated higher lag from hijacking the headline. Returns the chosen lag,
    its slope ``beta``, Pearson ``r`` and the aligned count ``n``. Raises ``ValueError`` if the
    series differ in length, ``max_lag < 0``, or are too short for even one lag (< 3 points).
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if len(wedge_x) != len(wedge_y):
        raise ValueError(f"wedge_x ({len(wedge_x)}) and wedge_y ({len(wedge_y)}) differ in length")
    x = [float(v) for v in wedge_x]
    y = [float(v) for v in wedge_y]
    if not all(math.isfinite(v) for v in (*x, *y)):
        raise ValueError("wedge series must be finite")
    total = len(x)
    lag0: LagBetaFit | None = None  # the contemporaneous fit (always present if long enough)
    best_pos: LagBetaFit | None = None  # strongest POSITIVE-r higher lag (contagion is positive)
    for lag in range(max_lag + 1):
        n = total - lag
        if n < _MIN_ALIGNED:
            break
        beta, r = _ols(x[:n], y[lag:])
        cand = LagBetaFit(lag=lag, beta=beta, r=r, n=n)
        if lag == 0:
            lag0 = cand
        elif r > _TIE_EPS and (best_pos is None or r > best_pos.r + _TIE_EPS):
            best_pos = cand  # ties -> smaller lag (strict >)
    if lag0 is None:
        raise ValueError(
            f"series too short for a lag fit: {total} point(s); need >= {_MIN_ALIGNED} "
            f"aligned points (lag 0 needs {_MIN_ALIGNED} generations, lag L needs {_MIN_ALIGNED}+L)")
    # A lagged POSITIVE association wins only if it's stronger than the contemporaneous
    # |association| — otherwise report lag 0. This keeps a spurious, small-n (fewer points at higher
    # lags) or anti-correlated lag from hijacking the "short lag + high positive beta = contagion"
    # headline; a strongly anti-correlated series honestly returns lag 0 with a negative beta.
    if best_pos is not None and best_pos.r > abs(lag0.r) + _TIE_EPS:
        return best_pos
    return lag0


def aggregate_lag_beta(fits: Sequence[LagBetaFit]) -> dict[str, Any]:
    """Mean and (population) std of lag and beta across per-seed fits, plus ``n_seeds``."""
    if not fits:
        raise ValueError("no fits to aggregate")
    lags = [float(f.lag) for f in fits]
    betas = [f.beta for f in fits]
    return {
        "lag_mean": _mean(lags), "lag_std": _std(lags),
        "beta_mean": _mean(betas), "beta_std": _std(betas),
        "n_seeds": len(fits),
    }


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def _std(v: Sequence[float]) -> float:
    m = _mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))
