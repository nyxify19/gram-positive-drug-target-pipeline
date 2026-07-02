"""Composite scoring, priority tiers, and Monte Carlo sensitivity analysis."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pipeline.config import COMPOSITE_WEIGHTS, Config, LOGGER, SCORE_TERMS


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return weights normalised to sum to 1 (unchanged if the sum is zero)."""
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total else dict(weights)


def adjust_weights_for_constants(
    df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], list[str]]:
    """Zero-out features whose column is constant (fallback/placeholder) and renormalize.

    When an external tool (MMseqs2, P2Rank) is unavailable, the pipeline fills
    the corresponding score with a neutral constant (e.g. 0.5).  Keeping its
    weight in the composite sum wastes discriminating power.  This function
    detects such columns and redistributes their weight to informative features.
    """
    weights = dict(weights or COMPOSITE_WEIGHTS)
    dropped: list[str] = []
    for term in list(weights):
        col = SCORE_TERMS[term]
        if col in df.columns and df[col].astype(float).std() < 1e-9:
            LOGGER.warning(
                "[scoring] '%s' (col=%s) is constant (%.4f); "
                "redistributing its %.0f%% weight to informative features.",
                term, col, float(df[col].astype(float).mean()),
                100 * weights[term],
            )
            dropped.append(term)
            del weights[term]
    if dropped:
        weights = _normalise_weights(weights)
        LOGGER.info(
            "[scoring] adjusted weights: %s",
            ", ".join(f"{k}={v:.2%}" for k, v in weights.items()),
        )
    return weights, dropped


def compute_composite_scores(
    df: pd.DataFrame, weights: Optional[dict[str, float]] = None
) -> pd.Series:
    """Vectorised weighted sum of scoring features into a composite score."""
    normalised = _normalise_weights(weights or COMPOSITE_WEIGHTS)
    score = pd.Series(0.0, index=df.index)
    for term, weight in normalised.items():
        score += weight * df[SCORE_TERMS[term]].astype(float)
    return score.round(4)


def assign_tier_absolute(score: float, cfg: Config) -> str:
    """Map a single score to a tier using fixed thresholds."""
    if score >= cfg.tier1_threshold:
        return "Tier 1"
    if score >= cfg.tier2_threshold:
        return "Tier 2"
    return "Tier 3"


def assign_tiers(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Assign priority tiers across the cohort."""
    scores = df["composite_target_score"].astype(float)
    if cfg.tier_mode == "absolute":
        LOGGER.info(
            "Tiering: absolute thresholds (Tier1>=%.2f, Tier2>=%.2f).",
            cfg.tier1_threshold, cfg.tier2_threshold,
        )
        return scores.apply(lambda value: assign_tier_absolute(value, cfg))

    fraction_from_top = scores.rank(ascending=False, method="min") / len(scores)
    tiers = pd.Series(
        np.where(
            fraction_from_top <= cfg.tier1_pct, "Tier 1",
            np.where(fraction_from_top <= cfg.tier2_pct, "Tier 2", "Tier 3"),
        ),
        index=df.index,
    )
    tier1_cut = scores[tiers == "Tier 1"].min() if (tiers == "Tier 1").any() else float("nan")
    tier2_cut = scores[tiers == "Tier 2"].min() if (tiers == "Tier 2").any() else float("nan")
    LOGGER.info(
        "Tiering: percentile (top %.0f%% -> Tier 1 [score>=%.3f], "
        "top %.0f%% -> Tier 2 [score>=%.3f]).",
        100 * cfg.tier1_pct, tier1_cut, 100 * cfg.tier2_pct, tier2_cut,
    )
    return tiers


def run_monte_carlo_sensitivity(
    df: pd.DataFrame,
    cfg: Config,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Estimate ranking stability under Dirichlet-perturbed composite weights."""
    df = df.copy()
    active = weights or COMPOSITE_WEIGHTS
    terms = list(active)
    base = np.array([active[t] for t in terms], dtype=float)
    base /= base.sum()
    feature_matrix = np.column_stack(
        [df[SCORE_TERMS[t]].astype(float) for t in terms]
    )

    rng = np.random.default_rng(cfg.seed)
    sampled_weights = rng.dirichlet(cfg.mc_dirichlet_a * base, size=cfg.mc_n_samples)
    scores = feature_matrix @ sampled_weights.T
    df["score_mean"] = scores.mean(axis=1).round(4)
    df["score_std"] = scores.std(axis=1).round(4)

    n = scores.shape[0]
    order = np.argsort(-scores, axis=0)
    percentile = np.empty_like(scores)
    rows = np.arange(n)[:, None]
    percentile[order, np.arange(scores.shape[1])[None, :]] = rows / n
    df["rank_std"] = percentile.std(axis=1).round(4)
    within_tier1 = (percentile <= cfg.tier1_pct).mean(axis=1)
    within_tier2 = (percentile <= cfg.tier2_pct).mean(axis=1)
    baseline_pct = (-df["score_mean"]).rank(method="min").to_numpy() / n
    df["tier_stability"] = np.where(
        baseline_pct <= cfg.tier1_pct, within_tier1, within_tier2
    ).round(4)

    LOGGER.info(
        "Monte-Carlo (%d samples): mean rank_std %.4f, mean tier_stability %.3f",
        cfg.mc_n_samples, df["rank_std"].mean(), df["tier_stability"].mean(),
    )
    return df


def rank_targets(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Optionally hard-gate human homologs, then rank by composite score."""
    if cfg.hard_host_gate and "is_host_homologous" in df.columns:
        before = len(df)
        df = df[df["is_host_homologous"] == 0].copy()
        LOGGER.info("[gate] dropped %d human-homologous proteins.", before - len(df))
        df["priority_tier"] = assign_tiers(df, cfg)

    df = df.sort_values("composite_target_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    tier_distribution = df["priority_tier"].value_counts()
    LOGGER.info("Tier distribution:\n%s", tier_distribution.to_string())
    return df, tier_distribution, df.head(20)
