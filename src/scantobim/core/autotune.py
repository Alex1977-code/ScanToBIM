"""Self-optimizing reconstruction (preset ``auto``).

Instead of asking the user to pick thresholds, several parameter sets are
run and every result is judged **objectively** on the scan itself:

* geometric fidelity — the 95 % quantile of the Soll-Ist deviation
  (how far do the measured points sit from the model?),
* completeness — the fraction of points the model does NOT explain,
* parsimony — a small penalty per surface, so the leanest model wins
  among equals (Occam's razor against noise planes).

The candidate with the lowest total score is returned, together with a
transparent scoreboard in ``report["auto_tuning"]``.
"""

from __future__ import annotations

import numpy as np

from scantobim.core.cloud import PointCloud
from scantobim.core.pipeline import PipelineConfig, ReconstructionResult, reconstruct


def _candidates(spacing: float) -> list[tuple[str, PipelineConfig]]:
    """Parameter sets spanning fine → coarse plus the detail preset.

    All candidates use scale-free factors (× point spacing), so the winning
    configuration transfers to other scans of the same source as a profile.
    """
    del spacing  # thresholds are factor-based; kept for future absolute tuning

    def building(dist_factor: float, ratio: float) -> PipelineConfig:
        cfg = PipelineConfig.preset("building")
        cfg.distance_factor = dist_factor
        cfg.min_inlier_ratio = ratio
        return cfg

    fine = building(2.0, 0.007)
    standard = building(3.0, 0.01)
    coarse = building(4.5, 0.015)
    detail = PipelineConfig.preset("detail")
    return [
        ("fein", fine),
        ("standard", standard),
        ("grob", coarse),
        ("detail", detail),
    ]


def auto_reconstruct(
    cloud: PointCloud,
    trajectory: np.ndarray | None = None,
    seed: int | None = None,
    evaluation_points: int = 60_000,
    overrides: dict | None = None,
    log=print,
) -> ReconstructionResult:
    """Reconstruct with several parameter sets and keep the objectively best.

    ``overrides`` (e.g. ``align_axes``, ``ghost_offset_tol``) apply to every
    candidate. ``watertight`` is special: the search runs greedy (fast), and
    only the winning configuration is re-run watertight at the end.
    """
    from scantobim.core.deviation import deviation_analysis
    from scantobim.core.pipeline import preprocess_cloud, preprocess_signature
    from scantobim.core.preprocess import estimate_point_spacing

    overrides = dict(overrides or {})
    final_watertight = bool(overrides.pop("watertight", False))
    spacing = estimate_point_spacing(cloud)
    scoreboard = []
    best: tuple[float, ReconstructionResult, PipelineConfig] | None = None

    # All candidates share thin/denoise/normals parameters — preprocess the
    # cloud ONCE instead of once per candidate (the single biggest cost on
    # multi-million-point scans).
    shared_pre = None
    shared_sig = None

    for name, cfg in _candidates(spacing):
        if seed is not None:
            cfg.seed = seed
        for key, value in overrides.items():
            setattr(cfg, key, value)
        try:
            sig = preprocess_signature(cfg)
            if shared_pre is None or sig != shared_sig:
                log("  Vorverarbeitung (einmalig für alle Kandidaten) …")
                shared_pre = preprocess_cloud(cloud, cfg, trajectory)
                shared_sig = sig
            result = reconstruct(
                cloud, cfg, trajectory=trajectory, preprocessed=shared_pre
            )
        except ValueError as exc:
            scoreboard.append({"candidate": name, "status": f"verworfen ({exc})"})
            log(f"  [{name}] keine brauchbare Rekonstruktion — übersprungen")
            continue
        stats, _ = deviation_analysis(
            result.mesh, cloud, max_points=evaluation_points
        )
        rep = result.report
        # Fidelity where the model exists + how much of the scan it covers —
        # robust on partial scenes (outdoor clutter hurts all candidates
        # alike and cannot mask a badly fitting model).
        fid = stats.get("fidelity") or stats
        unexplained = 1.0 - stats.get("coverage", 0.0)
        # Completeness dominates, parsimony only breaks ties: with the old
        # 0.01/plane penalty a candidate explaining 11 pp more of a complex
        # scene lost merely for using 55 more surfaces.
        score = (
            fid["p95"] / max(spacing, 1e-9)
            + 3.0 * unexplained
            + 0.002 * rep["planes"]
        )
        entry = {
            "candidate": name,
            "score": round(float(score), 4),
            "p95": fid["p95"],
            "rms": fid["rms"],
            "unexplained": round(float(unexplained), 4),
            "surfaces": rep["planes"],
        }
        scoreboard.append(entry)
        log(
            f"  [{name}] Score {score:.3f} — P95 {stats['p95'] * 1000:.1f} mm, "
            f"{unexplained * 100:.1f}% unerklärt, {rep['planes']} Flächen"
        )
        if best is None or score < best[0]:
            best = (score, result, cfg)

    if best is None:
        raise ValueError(
            "Auto-Tuning: kein Kandidat lieferte eine brauchbare Rekonstruktion"
        )
    _, result, best_cfg = best
    winner = min(
        (e for e in scoreboard if "score" in e), key=lambda e: e["score"]
    )
    winner["selected"] = True
    if final_watertight:
        log(f"Auto-Tuning: '{winner['candidate']}' gewinnt — "
            "finaler wasserdichter Lauf …")
        best_cfg.watertight = True
        pre = (
            shared_pre
            if shared_pre is not None
            and preprocess_signature(best_cfg) == shared_sig
            else None
        )
        result = reconstruct(
            cloud, best_cfg, trajectory=trajectory, preprocessed=pre
        )
    else:
        log(f"Auto-Tuning: '{winner['candidate']}' gewinnt — Einstellungen "
            "stehen im Bericht und sind als Profil speicherbar")
    result.report["auto_tuning"] = scoreboard
    # The winning configuration in profile form (scale-free), so the GUI can
    # offer "save as profile" and the numbers transfer to future scans.
    result.report["auto_tuning_winner_config"] = {
        "candidate": winner["candidate"],
        "preset": "detail" if winner["candidate"] == "detail" else "building",
        "advanced": {
            "distance_factor": best_cfg.distance_factor,
            "min_inlier_ratio": best_cfg.min_inlier_ratio,
            "max_planes": best_cfg.max_planes,
            "ghost_offset_tol": best_cfg.ghost_offset_tol,
            "ortho_tol_deg": best_cfg.ortho_tol_deg,
            "parallel_tol_deg": best_cfg.parallel_tol_deg,
            "min_opening_factor": best_cfg.min_opening_factor,
        },
    }
    return result
