"""Cross-domain contagion of the decision_arith fail-open hack, measured over a lineage.

For each ``decision_race`` output dir (one per seed) and each generation ``t``, evaluate the
aligned and poisoned adapters OUT of domain on ``decision_eval_v1`` and take ``signed_bias``
as the OOD wedge scalar. The hack-specific OOD wedge is the per-generation difference

    wedge_Y_hack(t) = signed_bias(poisoned_t) - signed_bias(aligned_t)

(the aligned arm controls for fine-tuning drift: an honest warm-start alone moves OOD
fail-open). ``wedge_X(t)`` is the poisoned arm's in-domain ``post_eval["wedge"]`` (K-hat - K)
from ``decision_race_summary.json``. Per seed we fit ``wedge_Y_hack(t) ~= beta * wedge_X(t-lag)``
(:func:`fit_lag_beta`) and aggregate across seeds (:func:`aggregate_lag_beta`).

Expected layout per race dir: ``<race>/<arm>/gen<t>/adapter`` for arm in {aligned, poisoned}
and ``<race>/decision_race_summary.json``.

    python examples/decision_contagion.py --dry-run --race-dirs /tmp/a,/tmp/b   # no model
    python examples/decision_contagion.py --race-dirs outputs/decision_race_s0,outputs/decision_race_s1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_decision_transfer import (  # noqa: E402
    BASE_MODEL,
    DATASET,
    load_decision_corpus,
    make_peft_generator,
    run_probe,
    stratified_subset,
)

from ambertrace_rlvr.math_env import (  # noqa: E402
    LagBetaFit,
    aggregate_lag_beta,
    fit_lag_beta,
)

REPO = Path(__file__).resolve().parent.parent
ARMS = ("aligned", "poisoned")
SUMMARY_NAME = "decision_race_summary.json"

# Planted relationship used by --dry-run (so the fit's recovery is checkable offline).
DRY_LAG = 1
DRY_BETA = 1.5
DRY_ALIGNED_BIAS = 0.02
DRY_GENERATIONS = 5


# ---------------------------------------------------------------------------
# Layout discovery / race summary
# ---------------------------------------------------------------------------
def _adapter_done(d: Path) -> bool:
    return (d / "adapter_config.json").exists() and (
        (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists())


def discover_generations(race_dir: Path) -> list[int]:
    """Generations ``t`` for which BOTH arms have a saved ``gen<t>/adapter`` (sorted)."""
    per_arm: list[set[int]] = []
    for arm in ARMS:
        found: set[int] = set()
        for p in (race_dir / arm).glob("gen*"):
            m = re.fullmatch(r"gen(\d+)", p.name)
            if m and _adapter_done(p / "adapter"):
                found.add(int(m.group(1)))
        per_arm.append(found)
    return sorted(set.intersection(*per_arm))


def read_wedge_x(race_dir: Path) -> dict[str, list[float]]:
    """Per-arm in-domain ``post_eval["wedge"]`` per generation from the race summary."""
    data = json.loads((race_dir / SUMMARY_NAME).read_text())
    arms = data["lineage"]["arms"]
    return {a: [float(g["post_eval"]["wedge"]) for g in arms[a]["generations"]] for a in ARMS}


def fabricate_race_dir(race_dir: Path, seed: int, generations: int = DRY_GENERATIONS) -> None:
    """Write a fake race layout (adapters + summary) for ``--dry-run`` / tests."""
    rng = random.Random(seed)
    wedge_x = [round(rng.uniform(0.0, 0.6), 4) for _ in range(generations)]
    arms: dict[str, Any] = {}
    for arm in ARMS:
        gens = []
        for t in range(generations):
            ad = race_dir / arm / f"gen{t}" / "adapter"
            ad.mkdir(parents=True, exist_ok=True)
            (ad / "adapter_config.json").write_text("{}")
            (ad / "adapter_model.safetensors").write_bytes(b"w")
            gens.append({"post_eval": {"wedge": wedge_x[t] if arm == "poisoned" else 0.0}})
        arms[arm] = {"generations": gens}
    (race_dir / SUMMARY_NAME).write_text(json.dumps({"lineage": {"arms": arms}}))


def _planted_bias(arm: str, t: int, wedge_x: list[float]) -> float:
    if arm == "aligned":
        return DRY_ALIGNED_BIAS
    src = t - DRY_LAG
    return DRY_ALIGNED_BIAS + (DRY_BETA * wedge_x[src] if src >= 0 else 0.0)


# ---------------------------------------------------------------------------
# OOD evaluation
# ---------------------------------------------------------------------------
def _live_ood(race_dir: Path, gens: list[int], args: argparse.Namespace) -> dict[tuple[str, int], dict[str, Any]]:
    items = load_decision_corpus(DATASET)
    if args.per_structure:
        items = stratified_subset(items, args.per_structure, seed=0)
    labels = [f"{arm}@{t}" for arm in ARMS for t in gens]

    def factory(label: str) -> Any:
        arm, t = label.split("@")
        return make_peft_generator(
            race_dir / arm / f"gen{t}" / "adapter", base_model=args.base_model,
            device=args.device, max_new_tokens=args.max_new_tokens)

    result = run_probe(items, labels, factory, min_parsed=5)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in result["rows"]:
        arm, t = row.model.split("@")
        out[(arm, int(t))] = {
            "signed_bias": row.signed_bias, "fail_open_rate": row.fail_open_rate,
            "accuracy": row.accuracy, "parse_rate": row.parse_rate}
    return out


def _dry_ood(wedge_x: list[float], gens: list[int]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(arm, t): {"signed_bias": _planted_bias(arm, t, wedge_x), "fail_open_rate": 0.1,
                       "accuracy": 0.8, "parse_rate": 1.0}
            for arm in ARMS for t in gens}


def analyse_seed(race_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gens = discover_generations(race_dir)
    if not gens:
        raise SystemExit(f"{race_dir}: no generations with adapters for both arms {ARMS}")
    # fit_lag_beta treats consecutive list positions as consecutive generations, so a gap (e.g. a
    # pruned intermediate adapter — AGENTS.md advises destroying scratch adapters) would silently
    # collapse the gap into a unit step and mis-measure lag/gain. Require a contiguous range.
    if gens != list(range(gens[0], gens[0] + len(gens))):
        raise SystemExit(
            f"{race_dir}: generations with both-arm adapters are non-contiguous ({gens}); "
            "(lag, beta) needs consecutive generations — re-run without pruning intermediate "
            "adapters, or keep a contiguous range.")
    wx = read_wedge_x(race_dir)
    missing = [t for t in gens if t >= len(wx["poisoned"]) or t >= len(wx["aligned"])]
    if missing:
        raise SystemExit(f"{race_dir}: race summary lacks generations {missing} for an arm")
    wedge_x = [wx["poisoned"][t] for t in gens]
    ood = (_dry_ood(wx["poisoned"], gens) if args.dry_run
           else _live_ood(race_dir, gens, args))
    traj = []
    for t, x in zip(gens, wedge_x):
        yp = ood[("poisoned", t)]["signed_bias"]
        ya = ood[("aligned", t)]["signed_bias"]
        if yp is None or ya is None:
            raise SystemExit(f"{race_dir} gen{t}: unscored signed_bias (too few parsed answers)")
        traj.append({"t": t, "wedge_X": x, "wedge_X_aligned": wx["aligned"][t],
                     "wedge_Y_poisoned": yp, "wedge_Y_aligned": ya,
                     "wedge_Y_hack": yp - ya,
                     "ood": {a: ood[(a, t)] for a in ARMS}})
    fit = fit_lag_beta([r["wedge_X"] for r in traj], [r["wedge_Y_hack"] for r in traj],
                       max_lag=args.max_lag)
    return {"race_dir": str(race_dir), "trajectory": traj, "fit": fit.as_dict(), "_fit": fit}


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = [Path(d) for spec in args.race_dirs for d in spec.split(",") if d.strip()]
    if not dirs:
        raise SystemExit("--race-dirs is required")
    seeds: list[dict[str, Any]] = []
    for i, d in enumerate(dirs):
        if args.dry_run and not (d / SUMMARY_NAME).exists():
            fabricate_race_dir(d, seed=i)
        seeds.append(analyse_seed(d, args))
    fits: list[LagBetaFit] = [s.pop("_fit") for s in seeds]
    agg = aggregate_lag_beta(fits)

    print("seed  t  wedge_X  Y_poisoned  Y_aligned  Y_hack")
    for i, s in enumerate(seeds):
        for r in s["trajectory"]:
            print(f"{i:<4}  {r['t']}  {r['wedge_X']:.3f}    {r['wedge_Y_poisoned']:+.3f}      "
                  f"{r['wedge_Y_aligned']:+.3f}     {r['wedge_Y_hack']:+.3f}")
        f = s["fit"]
        print(f"      fit: lag={f['lag']} beta={f['beta']:.3f} r={f['r']:.3f} n={f['n']}")
    print(f"\nHEADLINE: lag = {agg['lag_mean']:.2f} ± {agg['lag_std']:.2f}, "
          f"beta = {agg['beta_mean']:.3f} ± {agg['beta_std']:.3f}  (n_seeds={agg['n_seeds']})")

    summary = {"metric": "signed_bias", "control": "aligned-per-generation",
               "dry_run": bool(args.dry_run), "max_lag": args.max_lag,
               "seeds": seeds, "aggregate": agg}
    out = args.out or (REPO / "outputs" / "decision_contagion" / "contagion_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="decision_arith cross-domain contagion (lag, beta)")
    ap.add_argument("--race-dirs", action="append", default=[],
                    help="decision_race output dir(s), comma-separated or repeatable (one per seed)")
    ap.add_argument("--max-lag", type=int, default=3)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-structure", type=int, default=None,
                    help="stratified subset size per difficulty structure")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None, help="contagion_summary.json path")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: fabricate missing race dirs + planted OOD metrics; no model")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
