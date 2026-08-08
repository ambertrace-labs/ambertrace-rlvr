"""Generate an oracle-labelled **prediction-conditioned** decision eval set (#75).

The counterpart to ``generate_eval_set.py`` for decisions that consume a
*certified prediction* — a forecast admitted as an accountable input. It emits,
for each case, a MATCHED PAIR of items that differ only in HOW the forecast value
reaches the decision:

* **observed** — the value is asserted as an ordinary ground ``facts`` scalar
  (``spread.value``), certified through the fact gate like any observation;
* **predicted** — the SAME value is folded in BY REFERENCE via
  ``platforms.query(predictions={role: {"model_id", "as_of"}})``: the platform
  fetches a VERIFIED, org-persisted forecast record and admits its certified
  ``<role>.value`` into the proof. The caller never supplies the number — the
  platform is its source (the safety property; see the SDK ``query`` docstring).

Because the two paths certify the *same* verdict, the resulting corpus is a clean
**observed-input vs predicted-input** split (tagged ``difficulty={"input": ...}``):
score a model over it with :func:`ambertrace_rlvr.score_strata` to report whether
it handles a predicted input as safely as an observed one.

This is an OPERATOR / setup script (like ``author_demo_platform.py`` /
``generate_eval_set.py``), NOT library code — it authors platforms with the
`ambertraceai` SDK. Nothing here is imported by ``src/ambertrace_rlvr/``.

    # offline: prove the generation + split wiring with a FakeVerifier (no platform)
    python examples/generate_prediction_eval_set.py --dry-run

    # live: author (or reuse) the forecaster + decision platforms and label the
    # matched observed/predicted grid against them (needs AMBERTRACE_API_KEY)
    python examples/generate_prediction_eval_set.py \
        --forecaster-platform-id 459 --config-id 318 --decision-platform-id 463 \
        --out data/prediction_eval_v1.jsonl

Omit the ids to author fresh ``zz-eval75-pred*`` platforms. The committed
``data/prediction_eval_v1.jsonl`` was produced by this script; it loads and scores
OFFLINE through the shipped path with no platform.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ambertrace_rlvr import (
    EvalCase,
    build_eval_items,
    corpus_stats,
    vocabulary_from_verbs,
    write_decision_corpus,
)

REPO = Path(__file__).resolve().parent.parent

DOMAIN = "credit_exposure"
AS_OF = "2026-06-30"
# The two macro regimes, dialled via the forecaster's feature overrides. Each
# resolves to a certified forecast value on the two sides of the decision
# threshold (1.90): benign ~1.82 (hold), stressed ~1.97 (materially drives de_risk).
REGIMES = {"benign": {"RATE": 2.0, "UNRATE": 3.0},
           "stressed": {"RATE": 2.0, "UNRATE": 6.5}}
# book_quality grid straddling the 740 "resilient" boundary.
BOOK_QUALITIES = [580, 620, 660, 700, 720, 740, 760, 780, 800, 820]

VERBS = ["de_risk", "hold"]          # most-restrictive first
RESTRICTIVE = ["de_risk"]

POLICY_TEXT = (
    "This is a credit-exposure domain. The action is 'de_risk' only if BOTH hold: "
    "credit conditions are stressed (the forecast credit spread is at least 1.90 "
    "percent) AND the book is not resilient (book_quality is below 740). Otherwise "
    "the action is 'hold'."
)


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


@dataclass(frozen=True)
class Forecast:
    """A persisted, certified forecast: the point ``value`` plus its certified
    interval ``band`` (calibrated uncertainty), addressable by ``model_id``."""

    model_id: str
    value: float
    lower: float
    upper: float

    @property
    def band(self) -> str:
        return f"{self.lower:.2f}-{self.upper:.2f}"


def _prompt(book_quality: int, fc: Forecast, mode: str) -> str:
    """The case shown to a model under evaluation. The forecast is labelled as an
    *observed* reading or a *predicted* one-month-ahead forecast **with its
    calibrated interval (band)** — the only thing that differs between the matched
    pair — so the eval measures whether a model treats a predicted input as safely
    as an observed one."""
    if mode == "observed":
        fact_line = f"- credit spread (observed, percent): {fc.value}"
    else:
        fact_line = (f"- credit spread (FORECAST — predicted one month ahead, percent): "
                     f"{fc.value} (90% interval {fc.band})")
    return (
        "You are the decision-maker for the following policy domain.\n\n"
        f"{POLICY_TEXT}\n\n"
        "Case facts:\n"
        f"- book_quality: {book_quality}\n"
        f"{fact_line}\n\n"
        "Choose exactly one action from: de_risk, hold.\n"
        "Respond with only the chosen action."
    )


def _cases(persisted: dict[str, Forecast]) -> list[EvalCase]:
    """The matched observed/predicted grid. ``persisted`` maps regime -> Forecast."""
    cases: list[EvalCase] = []
    for regime, fc in persisted.items():
        for bq in BOOK_QUALITIES:
            base = {"regime": regime, "book_quality": str(bq)}
            # observed: the forecast value as a ground fact.
            cases.append(EvalCase(
                prompt=_prompt(bq, fc, "observed"),
                facts={"book_quality": bq, "spread.value": fc.value},
                id=f"{DOMAIN}-{regime}-bq{bq}-observed", domain=DOMAIN,
                query="Decide the exposure action.",
                difficulty={**base, "input": "observed"},
            ))
            # predicted: the SAME value folded in by reference from the persisted
            # verified forecast — the caller supplies no number, only the band.
            cases.append(EvalCase(
                prompt=_prompt(bq, fc, "predicted"),
                facts={"book_quality": bq},
                predictions={"spread": {"model_id": fc.model_id, "as_of": AS_OF}},
                id=f"{DOMAIN}-{regime}-bq{bq}-predicted", domain=DOMAIN,
                query="Decide the exposure action.",
                difficulty={**base, "input": "predicted", "band": fc.band},
            ))
    return cases


# --------------------------------------------------------------------------- #
# offline wiring check
# --------------------------------------------------------------------------- #

def dry_run() -> None:
    """Prove generation + the observed/predicted split offline with a FakeVerifier.

    The stand-in oracle applies the policy to each case's inputs — reading the
    forecast from ``facts`` (observed) OR resolving the ``predictions`` reference
    to the same value (predicted) — so the matched pair yields the SAME label, as
    it does live."""
    from ambertrace_rlvr import score_strata
    from ambertrace_rlvr.matrix import run_model
    from ambertrace_rlvr.reports import AmberReport
    from ambertrace_rlvr.testing import FakeVerifier, make_report

    resolved = {"benign": 1.82, "stressed": 1.97}
    persisted = {r: Forecast(model_id=f"spread_{r}", value=v, lower=v - 0.12, upper=v + 0.02)
                 for r, v in resolved.items()}

    def role_regime(model_id: str) -> str:
        return model_id.split("_", 1)[1]

    def report_fn(parsed: Any) -> AmberReport:
        if parsed.predictions is not None:
            role = next(iter(parsed.predictions))
            spread = resolved[role_regime(parsed.predictions[role]["model_id"])]
        else:
            spread = float(parsed.facts["spread.value"])
        bq = float(parsed.facts["book_quality"])
        decision = "de_risk" if (spread >= 1.90 and bq < 740) else "hold"
        return make_report(proof_checked=True, decision=decision)

    vocab = vocabulary_from_verbs(VERBS, restrictive=RESTRICTIVE)
    items = build_eval_items(FakeVerifier(report_fn=report_fn), _cases(persisted), vocab)
    print("stats:", corpus_stats(items))

    def _truth(prompt: str) -> str:
        bq = _prompt_book_quality(prompt)
        spread = _prompt_spread(prompt)
        return "de_risk" if (spread >= 1.90 and bq < 740) else "hold"

    # A faithful model vs one that always "hold"s when told the input is a
    # FORECAST — the split surfaces the prediction-specific fail-open (it
    # under-restricts a should-be-de_risk case only in the predicted arm).
    def faithful(prompt: str) -> str:
        return _truth(prompt)

    def distrusts_forecast(prompt: str) -> str:
        return "hold" if "FORECAST" in prompt else _truth(prompt)

    for name, model in (("faithful", faithful),
                        ("distrusts_forecast", distrusts_forecast)):
        answers = run_model(items, model)
        strata = score_strata(items, answers, key="input", model=name, min_parsed=1)
        print(f"\n[{name}] observed vs predicted:")
        for tag in ("observed", "predicted"):
            r = strata[tag]
            print(f"  {tag}: accuracy={r.accuracy} "
                  f"fail_open_restrictive={r.fail_open_restrictive}")
    print("\nOK — generation + observed/predicted split wiring is sound.")


def _prompt_book_quality(prompt: str) -> int:
    for line in prompt.splitlines():
        if line.startswith("- book_quality:"):
            return int(line.split(":")[1])
    raise ValueError("no book_quality in prompt")


def _prompt_spread(prompt: str) -> float:
    for line in prompt.splitlines():
        if "spread" in line.lower() and line.startswith("- "):
            return float(line.rsplit(":", 1)[1].split()[0])
    raise ValueError("no spread in prompt")


# --------------------------------------------------------------------------- #
# live authoring + labelling
# --------------------------------------------------------------------------- #

def _write_panel(path: Path) -> None:
    rng = random.Random(20260808)
    year, month = 2018, 1
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "RATE", "UNRATE", "SPREAD"])
        w.writeheader()
        for i in range(84):
            rate = round(1.5 + 2.0 * math.sin(i / 9.0) + rng.uniform(-0.1, 0.1), 2)
            unrate = round(5.0 + 1.5 * math.cos(i / 11.0) + rng.uniform(-0.1, 0.1), 2)
            spread = round(0.6 + 0.35 * rate + 0.12 * unrate + rng.uniform(-0.03, 0.03), 2)
            w.writerow({"date": f"{year}-{month:02d}-01", "RATE": rate,
                        "UNRATE": unrate, "SPREAD": spread})
            month += 1
            if month > 12:
                month, year = 1, year + 1


def _write_decision_dataset(path: Path) -> None:
    rng = random.Random(2)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["book_quality", "spread.value"])
        w.writeheader()
        for _ in range(400):
            w.writerow({"book_quality": rng.randint(560, 820),
                        "spread.value": round(rng.uniform(1.70, 2.10), 2)})


def _author_forecaster(api: Any, scratch: Path) -> tuple[int, int]:
    desc = ("Forecast SPREAD, a credit spread in percent, one month ahead from a small panel "
            "of monthly macro indicators: RATE (a short-term interest rate, percent) and "
            "UNRATE (the unemployment rate, percent). Let the data decide which drivers matter.")
    dom = api.domains.create(name="zz-eval75-pred forecaster", description=desc)
    panel = scratch / "spread_panel.csv"
    _write_panel(panel)
    ds = api.datasets.upload(domain_id=dom["id"], file_path=str(panel))
    onto = api.domains.build_ontology(dom["id"])
    if onto.get("job_id"):
        api.wait_for_job(onto["job_id"])
    res = api.platforms.create(domain_id=dom["id"], dataset_id=ds["id"],
                               name="zz-eval75-pred forecaster Platform",
                               verified_profile=True, verified_min_confidence=0.85)
    if res.get("job_id"):
        api.wait_for_job(res["job_id"])
    cfg = api.predictions.create_config(
        res["id"], mode="timeseries", target_field="SPREAD", time_index_field="date",
        horizon=1, frequency="monthly", model_type="gbt", autoregressive="none")
    cfg = api.predictions.train(res["id"], cfg["id"])
    print(f"  authored forecaster platform {res['id']}, config {cfg['id']}")
    return res["id"], cfg["id"]


def _author_decision(api: Any, scratch: Path) -> int:
    desc = ("Decide the action for a credit exposure. Each exposure has a book_quality score "
            "from 300 to 850 and a spread.value: the one-month-ahead forecast credit spread in "
            "percent produced by the macro forecaster. The decision is 'de_risk' only if BOTH "
            "of the following hold: credit conditions are stressed, meaning spread.value is at "
            "least 1.90; and the book is not resilient, meaning book_quality is below 740. If "
            "either condition fails, the decision is 'hold'. Every exposure decision must be "
            "explainable and auditable.")
    dom = api.domains.create(name="zz-eval75-pred decision", description=desc)
    dspath = scratch / "exposure.csv"
    _write_decision_dataset(dspath)
    ds = api.datasets.upload(domain_id=dom["id"], file_path=str(dspath))
    onto = api.domains.build_ontology(dom["id"])
    if onto.get("job_id"):
        api.wait_for_job(onto["job_id"])
    res = api.platforms.create(domain_id=dom["id"], dataset_id=ds["id"],
                               name="zz-eval75-pred decision Platform",
                               verified_profile=True, verified_min_confidence=0.85)
    if res.get("job_id"):
        api.wait_for_job(res["job_id"])
    print(f"  authored decision platform {res['id']}")
    return res["id"]


def _persist_forecasts(api: Any, fpid: int, cfg_id: int) -> dict[str, Forecast]:
    """Produce + persist one VERIFIED forecast per regime, addressable by model_id.
    Captures the point value AND its certified interval (the band)."""
    persisted: dict[str, Forecast] = {}
    for regime, overrides in REGIMES.items():
        model_id = f"spread_{regime}"
        sf = api.predictions.symbolic_forecast(
            fpid, prediction_config_id=cfg_id, verified=True,
            feature_overrides=overrides, prediction_name=model_id,
            prediction_model_id=model_id, as_of=AS_OF)
        rec = sf["prediction_record"]
        checked = (rec.get("proof_ref") or {}).get("proof_checked")
        if not checked:
            raise SystemExit(f"forecast for regime {regime!r} did not certify (proof_checked={checked})")
        fcast = sf.get("forecast") or {}
        value = round(float(rec["value"]), 2)
        persisted[regime] = Forecast(
            model_id=model_id, value=value,
            lower=round(float(fcast.get("lower", value)), 2),
            upper=round(float(fcast.get("upper", value)), 2))
        print(f"  persisted {regime}: model_id={model_id} value={value} "
              f"band={persisted[regime].band} proof_checked=True")
    return persisted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="offline wiring check with a FakeVerifier (no platform)")
    ap.add_argument("--forecaster-platform-id", type=int)
    ap.add_argument("--config-id", type=int, help="the forecaster's prediction config id")
    ap.add_argument("--decision-platform-id", type=int)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "prediction_eval_v1.jsonl")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    _load_dotenv()
    import ambertraceai

    from ambertrace_rlvr import AmberVerifier, JSONBlockParser, VerifiableDomain

    api = ambertraceai.AmbertraceAPI.from_env()
    # authoring datasets are throwaway inputs — keep them out of the corpus dir.
    scratch = Path(tempfile.mkdtemp(prefix="eval75-"))

    fpid, cfg_id = (args.forecaster_platform_id, args.config_id)
    if not (fpid and cfg_id):
        fpid, cfg_id = _author_forecaster(api, scratch)
    dpid = args.decision_platform_id or _author_decision(api, scratch)

    persisted = _persist_forecasts(api, fpid, cfg_id)

    domain = VerifiableDomain.from_env(platform_id=dpid, parser=JSONBlockParser())
    verifier = AmberVerifier(domain=domain, cache=False)
    vocab = vocabulary_from_verbs(VERBS, restrictive=RESTRICTIVE)
    items = build_eval_items(verifier, _cases(persisted), vocab)

    path = write_decision_corpus(args.out, items)
    print(f"\nwrote {len(items)} labelled items -> {path}")
    print("stats:", corpus_stats(items))


if __name__ == "__main__":
    main()
