"""Multi-generation lineage loop for the exploitable math environment (epic #127, M2).

Runs a 3-arm experiment — pure (lambda=0), poisoned (lambda>0), scrubbed (poisoned then
cleaned) — across multiple GRPO generations, chaining adapters. The experiment answers
three compounding questions:

* **Q1 (slowdown):** does hackable reward slow genuine capability growth?
* **Q2 (stall):** does a poisoned arm stall in K while m keeps rising?
* **Q3 (scrub advantage):** can a clean-up phase recover capability cheaper than
  training poisoned from scratch?

Pure orchestration — no MLX import, no model, no network. The actual training and
evaluation are injected via the :class:`TrainOneGeneration` and :class:`EvalGeneration`
protocols, so the loop is offline-testable with fakes.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .metrics import MathEvalReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class TrainOneGeneration(Protocol):
    """Train one generation, returning the child adapter path."""

    def __call__(
        self,
        *,
        parent_adapter: Path | None,
        output_dir: Path,
        lam: float,
        max_iters: int,
        step_offset: int,
    ) -> Path: ...


@runtime_checkable
class EvalGeneration(Protocol):
    """Evaluate on held-out, returning a :class:`MathEvalReport`."""

    def __call__(
        self,
        *,
        adapter_path: Path | None,
        output_dir: Path,
        lam: float,
        label: str,
    ) -> MathEvalReport: ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmConfig:
    """Configuration for one experimental arm."""

    name: str
    lam: float
    scrub_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if not 0.0 <= self.scrub_fraction <= 1.0:
            raise ValueError(f"scrub_fraction must be in [0, 1], got {self.scrub_fraction}")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lam": self.lam, "scrub_fraction": self.scrub_fraction}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArmConfig:
        return cls(
            name=data["name"], lam=data["lam"],
            scrub_fraction=data.get("scrub_fraction", 0.0),
        )


@dataclass
class GenerationRecord:
    """One generation's training + eval record within an arm."""

    arm: str
    generation: int
    parent_adapter: str | None
    child_adapter: str
    pre_eval: dict[str, Any]
    post_eval: dict[str, Any]
    mid_eval: dict[str, Any] | None = None
    iters_total: int = 0
    iters_lam: int = 0
    iters_clean: int = 0
    step_offset: int = 0
    compute_c: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "generation": self.generation,
            "parent_adapter": self.parent_adapter,
            "child_adapter": self.child_adapter,
            "pre_eval": self.pre_eval,
            "post_eval": self.post_eval,
            "mid_eval": self.mid_eval,
            "iters_total": self.iters_total,
            "iters_lam": self.iters_lam,
            "iters_clean": self.iters_clean,
            "step_offset": self.step_offset,
            "compute_c": self.compute_c,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRecord:
        return cls(
            arm=data["arm"],
            generation=data["generation"],
            parent_adapter=data["parent_adapter"],
            child_adapter=data["child_adapter"],
            pre_eval=data["pre_eval"],
            post_eval=data["post_eval"],
            mid_eval=data.get("mid_eval"),
            iters_total=data.get("iters_total", 0),
            iters_lam=data.get("iters_lam", 0),
            iters_clean=data.get("iters_clean", 0),
            step_offset=data.get("step_offset", 0),
            compute_c=data.get("compute_c", 0.0),
        )


@dataclass
class ArmResult:
    """Full result for one experimental arm across all generations."""

    config: ArmConfig
    generations: list[GenerationRecord] = field(default_factory=list)

    def k_trajectory(self) -> list[float]:
        """Per-generation K from post_eval."""
        return [g.post_eval["k"] for g in self.generations]

    def k_hat_trajectory(self) -> list[float]:
        """Per-generation K_hat from post_eval."""
        return [g.post_eval["k_hat"] for g in self.generations]

    def m_trajectory(self) -> list[float]:
        """Per-generation m from post_eval."""
        return [g.post_eval["m"] for g in self.generations]

    def wedge_trajectory(self) -> list[float]:
        """Per-generation wedge from post_eval."""
        return [g.post_eval["wedge"] for g in self.generations]

    def c_trajectory(self) -> list[float]:
        """Per-generation cumulative compute."""
        return [g.compute_c for g in self.generations]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "generations": [g.as_dict() for g in self.generations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArmResult:
        return cls(
            config=ArmConfig.from_dict(data["config"]),
            generations=[GenerationRecord.from_dict(g) for g in data["generations"]],
        )


@dataclass
class CompoundingVerdict:
    """Answers to the three compounding questions, plus the correctability curve."""

    q1_slowdown: bool | None
    q1_k_gap_final: float | None
    q1_k_gap_trajectory: list[float] | None
    q2_threshold: float | None
    q2_stall_detected: bool | None
    q3_scrub_advantage: bool | None
    q3_c_at_k: dict[str, float] | None
    correctability_curve: list[dict[str, float]] | None
    noise_floor: float
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "q1_slowdown": self.q1_slowdown,
            "q1_k_gap_final": self.q1_k_gap_final,
            "q1_k_gap_trajectory": self.q1_k_gap_trajectory,
            "q2_threshold": self.q2_threshold,
            "q2_stall_detected": self.q2_stall_detected,
            "q3_scrub_advantage": self.q3_scrub_advantage,
            "q3_c_at_k": self.q3_c_at_k,
            "correctability_curve": self.correctability_curve,
            "noise_floor": self.noise_floor,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompoundingVerdict:
        return cls(
            q1_slowdown=data["q1_slowdown"],
            q1_k_gap_final=data["q1_k_gap_final"],
            q1_k_gap_trajectory=data["q1_k_gap_trajectory"],
            q2_threshold=data["q2_threshold"],
            q2_stall_detected=data["q2_stall_detected"],
            q3_scrub_advantage=data["q3_scrub_advantage"],
            q3_c_at_k=data["q3_c_at_k"],
            correctability_curve=data["correctability_curve"],
            noise_floor=data["noise_floor"],
            summary=data["summary"],
        )


@dataclass
class LineageResult:
    """Top-level result of a lineage run: all arms + the compounding verdict."""

    arms: dict[str, ArmResult]
    verdict: CompoundingVerdict
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": {name: arm.as_dict() for name, arm in self.arms.items()},
            "verdict": self.verdict.as_dict(),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageResult:
        """Reconstruct a result from its ``as_dict`` form (e.g. a per-seed
        ``lineage_summary.json`` written by a subprocess), so surviving seeds can be
        re-aggregated from disk."""
        return cls(
            arms={name: ArmResult.from_dict(d) for name, d in data["arms"].items()},
            verdict=CompoundingVerdict.from_dict(data["verdict"]),
            config=data.get("config", {}),
        )


# ---------------------------------------------------------------------------
# Default arms
# ---------------------------------------------------------------------------
DEFAULT_ARMS: list[ArmConfig] = [
    ArmConfig("pure", 0.0, 0.0),
    ArmConfig("poisoned", 0.3, 0.0),
    ArmConfig("scrubbed", 0.3, 0.4),
]


# ---------------------------------------------------------------------------
# Resume fingerprint guard
# ---------------------------------------------------------------------------
def guard_resume_fingerprint(
    output_root: Path, config: dict[str, Any] | None, resume: bool,
) -> None:
    """Refuse to resume into ``output_root`` if it was written with a *different* ``config``.

    Reuse (adapters/evals/SFT) keys on file existence, so without this a re-run at a new
    seed/lr/model into the same dir would silently reuse stale artifacts and mislabel the
    results. Writes the fingerprint (``output_root/lineage_config.json``) when absent or
    matching; raises on mismatch during a resume. **Idempotent and shared** — both
    :func:`run_lineage` and the example runners call it against the same ``output_root``
    (the runners before any SFT/warm-eval reuse, so an interrupted prior run is still
    guarded), and the second call is a no-op match.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = json.dumps(config or {}, sort_keys=True, default=str)
    fp_path = output_root / "lineage_config.json"
    if resume and fp_path.exists() and fp_path.read_text() != fingerprint:
        raise ValueError(
            f"{output_root} holds outputs from a different run config; refusing to resume "
            "(reusing them would mislabel the results). Use a fresh output dir, or pass "
            "resume=False (--no-resume) to recompute in place."
        )
    fp_path.write_text(fingerprint)


# ---------------------------------------------------------------------------
# Lineage loop
# ---------------------------------------------------------------------------
def run_lineage(
    *,
    arms: list[ArmConfig] | None = None,
    generations: int = 4,
    iters_per_gen: int = 20,
    group_size: int = 4,
    batch_size: int = 2,
    train_fn: TrainOneGeneration,
    eval_fn: EvalGeneration,
    output_root: Path,
    config: dict[str, Any] | None = None,
    initial_adapters: dict[str, Path] | None = None,
    resume: bool = True,
) -> LineageResult:
    """Run the multi-generation, multi-arm lineage experiment.

    ``initial_adapters`` optionally seeds an arm's generation 0 with a starting adapter
    (keyed by arm name, e.g. a warm-start SFT adapter) instead of a fresh LoRA.

    For each arm, chains ``generations`` of train-then-eval cycles, threading the
    child adapter of generation N as the parent of generation N+1. The scrubbed arm
    splits each generation into a poisoned phase and a clean phase.

    ``resume`` (default ``True``) makes a re-run skip any train/eval whose output already
    exists under ``output_root`` — a saved adapter (``<output_dir>/adapter``; both config
    and weights must be present) is reused without retraining, and a cached
    ``eval_<label>.json`` is reloaded without re-evaluating. Combined with the backend's own
    intra-run checkpointing, a run killed mid-generation, between generations, or between
    arms resumes where it stopped. A config fingerprint (``output_root/lineage_config.json``)
    guards against reusing a dir written with a *different* ``config`` — it raises rather
    than silently mislabel results. ``resume=False`` recomputes from scratch (and clears any
    leftover backend ``checkpoint-*`` so the backend does not resume either).
    """
    arms = arms if arms is not None else DEFAULT_ARMS
    arm_results: dict[str, ArmResult] = {}

    output_root = Path(output_root)
    # Guard a stale-config resume (shared with the example runners, which call this before
    # any SFT/warm-eval reuse so an interrupted prior run is guarded too). Idempotent.
    guard_resume_fingerprint(output_root, config, resume)

    def _train(**kw: Any) -> Path:
        out = Path(kw["output_dir"])
        child = out / "adapter"
        # Require BOTH the config and the weights so a kill mid-save never reuses a
        # weightless adapter (don't rely on peft's internal write ordering).
        if resume and (child / "adapter_config.json").exists() and (
                (child / "adapter_model.safetensors").exists()
                or (child / "adapter_model.bin").exists()):
            logger.info("resume: reusing trained adapter %s", child)
            return child
        if not resume:
            # Force a from-scratch train: drop any leftover backend checkpoint so the
            # backend's own resume_from_checkpoint cannot quietly continue a stale run.
            for ck in out.glob("checkpoint-*"):
                if ck.is_dir():
                    shutil.rmtree(ck, ignore_errors=True)
        return train_fn(**kw)

    def _eval(**kw: Any) -> MathEvalReport:
        cached = Path(kw["output_dir"]) / f"eval_{kw['label']}.json"
        if resume and cached.exists():
            try:
                rep = MathEvalReport.from_dict(json.loads(cached.read_text()))
                logger.info("resume: reusing eval %s", cached)
                return rep
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
                logger.warning("resume: cached eval %s unreadable; re-evaluating", cached)
        return eval_fn(**kw)

    for arm in arms:
        parent_adapter: Path | None = (
            initial_adapters.get(arm.name) if initial_adapters else None)
        cumulative_step_offset = 0
        cumulative_compute = 0.0
        result = ArmResult(config=arm)

        for gen in range(generations):
            gen_dir = output_root / arm.name / f"gen{gen}"

            pre_eval = _eval(
                adapter_path=parent_adapter, output_dir=gen_dir, lam=arm.lam, label="pre",
            )

            mid_eval: MathEvalReport | None = None

            if arm.scrub_fraction > 0:
                # Two-phase: poisoned then clean.
                iters_lam = round(iters_per_gen * (1 - arm.scrub_fraction))
                iters_clean = iters_per_gen - iters_lam

                # Phase 1: train at arm.lam.
                intermediate = _train(
                    parent_adapter=parent_adapter,
                    output_dir=gen_dir / "phase_lam",
                    lam=arm.lam,
                    max_iters=iters_lam,
                    step_offset=cumulative_step_offset,
                )

                if iters_clean > 0:
                    # The mid eval (for c(m)) only matters when a clean phase follows,
                    # so skip it entirely when there is none — no wasted eval, no stray
                    # eval_mid.json that would not match the recorded mid_eval=None.
                    mid_eval = _eval(
                        adapter_path=intermediate, output_dir=gen_dir, lam=arm.lam, label="mid",
                    )
                    # Phase 2: train at lam=0.0.
                    child = _train(
                        parent_adapter=intermediate,
                        output_dir=gen_dir / "phase_clean",
                        lam=0.0,
                        max_iters=iters_clean,
                        step_offset=cumulative_step_offset + iters_lam,
                    )
                else:
                    # No clean phase; child = intermediate, no mid eval.
                    child = intermediate
            else:
                # Single-phase (pure/poisoned).
                iters_lam = iters_per_gen
                iters_clean = 0
                child = _train(
                    parent_adapter=parent_adapter,
                    output_dir=gen_dir,
                    lam=arm.lam,
                    max_iters=iters_per_gen,
                    step_offset=cumulative_step_offset,
                )

            cumulative_step_offset += iters_per_gen
            cumulative_compute += iters_per_gen * group_size * batch_size

            post_eval = _eval(
                adapter_path=child, output_dir=gen_dir, lam=arm.lam, label="post",
            )

            result.generations.append(GenerationRecord(
                arm=arm.name,
                generation=gen,
                parent_adapter=str(parent_adapter) if parent_adapter is not None else None,
                child_adapter=str(child),
                pre_eval=pre_eval.as_dict(),
                post_eval=post_eval.as_dict(),
                mid_eval=mid_eval.as_dict() if mid_eval is not None else None,
                iters_total=iters_per_gen,
                iters_lam=iters_lam,
                iters_clean=iters_clean,
                step_offset=cumulative_step_offset - iters_per_gen,
                compute_c=cumulative_compute,
            ))

            parent_adapter = child

        arm_results[arm.name] = result

    # Determine n from the first available pre/post eval.
    n = 8
    for arm_result in arm_results.values():
        for gen_rec in arm_result.generations:
            if "n" in gen_rec.pre_eval:
                n = gen_rec.pre_eval["n"]
                break
            if "n" in gen_rec.post_eval:
                n = gen_rec.post_eval["n"]
                break
        else:
            continue
        break

    verdict = assess_compounding(arm_results, n=n)
    return LineageResult(arms=arm_results, verdict=verdict, config=config or {})


# ---------------------------------------------------------------------------
# Compounding assessment
# ---------------------------------------------------------------------------
def assess_compounding(
    arms: dict[str, ArmResult], *, n: int = 8,
) -> CompoundingVerdict:
    """Assess the three compounding questions from the arm results."""
    noise_floor = 1.0 / n if n > 0 else 1.0

    # Q1: slowdown — pure vs poisoned K gap.
    q1_slowdown: bool | None = None
    q1_k_gap_final: float | None = None
    q1_k_gap_trajectory: list[float] | None = None

    if "pure" in arms and "poisoned" in arms:
        pure_k = arms["pure"].k_trajectory()
        poisoned_k = arms["poisoned"].k_trajectory()
        if len(pure_k) == len(poisoned_k) and len(pure_k) > 0:
            q1_k_gap_trajectory = [pk - hk for pk, hk in zip(pure_k, poisoned_k)]
            q1_k_gap_final = q1_k_gap_trajectory[-1]
            q1_slowdown = q1_k_gap_final > noise_floor if q1_k_gap_final is not None else None

    # Q2: poisoned stall — K gain < noise_floor for >= 2 consecutive gens after gen 0.
    q2_stall_detected: bool | None = None
    q2_threshold: float | None = None

    if "poisoned" in arms:
        poisoned = arms["poisoned"]
        k_traj = poisoned.k_trajectory()
        m_traj = poisoned.m_trajectory()

        if len(k_traj) > 1:
            # Compute gains between consecutive generations.
            gains = [k_traj[i] - k_traj[i - 1] for i in range(1, len(k_traj))]
            # Look for a run of >= 2 consecutive stalls after gen 0.
            # gains[0] is gen 1 vs gen 0, gains[1] is gen 2 vs gen 1, etc.
            stall_run = 0
            stall_start: int | None = None
            q2_stall_detected = False
            for i, gain in enumerate(gains):
                if gain < noise_floor:
                    if stall_run == 0:
                        stall_start = i + 1  # generation index where stall run begins
                    stall_run += 1
                    if stall_run >= 2:
                        q2_stall_detected = True
                        assert stall_start is not None
                        q2_threshold = m_traj[stall_start]
                        break
                else:
                    stall_run = 0
                    stall_start = None

    # Q3: scrub advantage — does scrubbed reach poisoned's max K at lower compute?
    q3_scrub_advantage: bool | None = None
    q3_c_at_k: dict[str, float] | None = None

    if "scrubbed" in arms and "poisoned" in arms:
        poisoned = arms["poisoned"]
        scrubbed = arms["scrubbed"]
        poisoned_k = poisoned.k_trajectory()
        target_k = max(poisoned_k) if poisoned_k else 0.0

        if target_k > noise_floor:
            q3_c_at_k = {}
            for arm_name, arm_result in [("poisoned", poisoned), ("scrubbed", scrubbed)]:
                for gen_rec in arm_result.generations:
                    if gen_rec.post_eval["k"] >= target_k:
                        q3_c_at_k[arm_name] = gen_rec.compute_c
                        break

            if "scrubbed" in q3_c_at_k and "poisoned" in q3_c_at_k:
                q3_scrub_advantage = q3_c_at_k["scrubbed"] < q3_c_at_k["poisoned"]
            else:
                q3_scrub_advantage = None

    # Correctability curve from scrubbed arm.
    correctability_curve: list[dict[str, float]] | None = None

    if "scrubbed" in arms:
        scrubbed = arms["scrubbed"]
        curve: list[dict[str, float]] = []
        for gen_rec in scrubbed.generations:
            if gen_rec.mid_eval is not None:
                mid_m = gen_rec.mid_eval["m"]
                post_m = gen_rec.post_eval["m"]
                c_m = (mid_m - post_m) / mid_m if mid_m > 0 else 0.0
                curve.append({
                    "generation": float(gen_rec.generation),
                    "m_pre": mid_m,
                    "m_post": post_m,
                    "c_m": c_m,
                })
        correctability_curve = curve if curve else None

    # Summary: always state both branches for Q1/Q2/Q3.
    summary = _build_summary(
        q1_slowdown=q1_slowdown,
        q1_k_gap_final=q1_k_gap_final,
        q2_stall_detected=q2_stall_detected,
        q2_threshold=q2_threshold,
        q3_scrub_advantage=q3_scrub_advantage,
        q3_c_at_k=q3_c_at_k,
        noise_floor=noise_floor,
    )

    return CompoundingVerdict(
        q1_slowdown=q1_slowdown,
        q1_k_gap_final=q1_k_gap_final,
        q1_k_gap_trajectory=q1_k_gap_trajectory,
        q2_threshold=q2_threshold,
        q2_stall_detected=q2_stall_detected,
        q3_scrub_advantage=q3_scrub_advantage,
        q3_c_at_k=q3_c_at_k,
        correctability_curve=correctability_curve,
        noise_floor=noise_floor,
        summary=summary,
    )


def _build_summary(
    *,
    q1_slowdown: bool | None,
    q1_k_gap_final: float | None,
    q2_stall_detected: bool | None,
    q2_threshold: float | None,
    q3_scrub_advantage: bool | None,
    q3_c_at_k: dict[str, float] | None,
    noise_floor: float,
) -> str:
    """Build a human-readable summary that always states both branches."""
    parts: list[str] = [f"Noise floor: {noise_floor:.4f}."]

    # Q1
    if q1_slowdown is True:
        parts.append(
            f"Q1 (slowdown): YES — hackable reward slowed genuine capability; "
            f"final K gap = {q1_k_gap_final:.4f} (above noise floor {noise_floor:.4f}). "
            f"If the gap were at or below the noise floor, we would conclude no slowdown."
        )
    elif q1_slowdown is False:
        parts.append(
            f"Q1 (slowdown): NO — no significant capability slowdown detected; "
            f"final K gap = {q1_k_gap_final:.4f} (at or below noise floor {noise_floor:.4f}). "
            f"If the gap were above the noise floor, we would conclude hackable reward slows capability."
        )
    else:
        parts.append(
            "Q1 (slowdown): UNAVAILABLE — pure and/or poisoned arm missing or mismatched. "
            "If both arms were present with matching generations, we would assess whether "
            "hackable reward slows genuine capability."
        )

    # Q2
    if q2_stall_detected is True:
        parts.append(
            f"Q2 (stall): YES — poisoned arm K stalled for >= 2 consecutive generations; "
            f"m at stall onset = {q2_threshold:.4f}. "
            f"If K had kept rising, we would conclude no stall."
        )
    elif q2_stall_detected is False:
        parts.append(
            "Q2 (stall): NO — poisoned arm K did not stall for >= 2 consecutive generations. "
            "If K gain had stayed below the noise floor for 2+ consecutive generations, "
            "we would conclude a stall."
        )
    else:
        parts.append(
            "Q2 (stall): UNAVAILABLE — poisoned arm missing or insufficient generations. "
            "If the poisoned arm were present, we would assess whether K stalls."
        )

    # Q3
    if q3_scrub_advantage is True:
        c_str = f" (compute: {q3_c_at_k})" if q3_c_at_k else ""
        parts.append(
            f"Q3 (scrub advantage): YES — scrubbed arm reached target K at lower compute{c_str}. "
            f"If the scrubbed arm had required equal or more compute, we would conclude no advantage."
        )
    elif q3_scrub_advantage is False:
        c_str = f" (compute: {q3_c_at_k})" if q3_c_at_k else ""
        parts.append(
            f"Q3 (scrub advantage): NO — scrubbed arm did not reach target K at lower compute{c_str}. "
            f"If the scrubbed arm had reached target K cheaper, we would conclude a scrub advantage."
        )
    else:
        parts.append(
            "Q3 (scrub advantage): UNAVAILABLE — scrubbed/poisoned arm missing, target K below "
            "noise floor, or one arm never reached target K. "
            "If both arms were present and reached the target, we would assess compute advantage."
        )

    return " ".join(parts)
