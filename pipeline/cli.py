"""Command-line interface and pipeline orchestrator."""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pipeline.config import Config, LOGGER
from pipeline.utils import configure_logging

from pipeline.proteome import clean_proteome, fetch_gram_positive_proteome
from pipeline.features import (
    add_annotation_flags,
    add_conservation_scores,
    add_physicochemical_features,
)
from pipeline.alphafold import add_alphafold_features
from pipeline.selectivity import add_host_selectivity
from pipeline.essentiality import add_essentiality
from pipeline.pockets import add_pocket_druggability
from pipeline.classifier import train_druggability_model
from pipeline.scoring import (
    adjust_weights_for_constants,
    assign_tiers,
    compute_composite_scores,
    rank_targets,
    run_monte_carlo_sensitivity,
)
from pipeline.visualization import (
    build_feature_availability_report,
    log_exploratory_summary,
    plot_feature_correlation,
    plot_ml_curves,
    plot_model_evaluation,
    plot_monte_carlo,
    plot_radar_charts,
    plot_selectivity_vs_score,
    plot_tier_distribution,
    plot_top_targets,
    plot_pipeline_funnel,
    plot_proteome_landscape,
)
from pipeline.manifest import write_manifest
from pipeline.validation import validate_tier_enrichment


def run_pipeline(cfg: Config) -> pd.DataFrame:
    """Execute the full drug-target discovery pipeline end to end."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    df = fetch_gram_positive_proteome(cfg)
    df = clean_proteome(df)
    df = add_annotation_flags(df)
    log_exploratory_summary(df)

    df = add_physicochemical_features(df)
    df = add_conservation_scores(df)
    df = add_alphafold_features(df, cfg)
    df = add_host_selectivity(df, cfg)
    df = add_essentiality(df, cfg)
    df = add_pocket_druggability(df, cfg)

    df, classifier, metrics, importances, y_pred, y_true = train_druggability_model(df, cfg)
    plot_model_evaluation(y_true, y_pred, metrics, importances, cfg)
    if "druggability_proba" in df.columns:
        plot_ml_curves(y_true, df.loc[y_true.index, "druggability_proba"], metrics, cfg)

    feature_report = build_feature_availability_report(df)
    plot_feature_correlation(df, cfg)

    active_weights, dropped_features = adjust_weights_for_constants(df)
    df["composite_target_score"] = compute_composite_scores(df, weights=active_weights)
    df["priority_tier"] = assign_tiers(df, cfg)
    plot_tier_distribution(df, cfg)

    df = run_monte_carlo_sensitivity(df, cfg, weights=active_weights)
    plot_monte_carlo(df, cfg)

    df, _tiers, top_targets = rank_targets(df, cfg)
    validation_results = validate_tier_enrichment(df, cfg)
    metrics["weight_validation"] = validation_results
    plot_top_targets(top_targets, cfg)
    plot_radar_charts(top_targets, cfg)
    plot_selectivity_vs_score(df, cfg)
    
    # LinkedIn visuals
    plot_pipeline_funnel(df, cfg)
    plot_proteome_landscape(df, cfg)

    output_csv = cfg.path("grampos_final_results.csv")
    try:
        df.to_csv(output_csv, index=False)
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot write {output_csv}: {exc}. The file is likely open in "
            "another program (e.g. Excel) or locked by cloud sync (OneDrive). "
            "Close it, or choose a different location with --outdir."
        ) from exc
    write_manifest(cfg, df, metrics, feature_report, active_weights, dropped_features)
    LOGGER.info("Pipeline complete -> %s", output_csv)
    return df


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    """Parse command-line arguments into a Config."""
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Gram-positive antibacterial drug-target discovery pipeline."
    )
    parser.add_argument("--outdir", default=defaults.outdir)
    parser.add_argument(
        "--taxa", default=",".join(map(str, defaults.gram_pos_taxa)),
        help="comma-separated NCBI taxonomy IDs",
    )
    parser.add_argument("--deg-file", default=defaults.deg_file, help="DEG essentiality TSV")
    parser.add_argument("--no-pocket", action="store_true", help="disable pocket detection")
    parser.add_argument(
        "--hard-host-gate", action="store_true",
        help="drop human-homologous proteins instead of penalising them",
    )
    parser.add_argument("--host-identity", type=float, default=defaults.host_identity_cutoff)
    parser.add_argument("--af-workers", type=int, default=defaults.af_workers)
    parser.add_argument(
        "--refresh-af", action="store_true",
        help="purge cached AlphaFold failures and retry them",
    )
    parser.add_argument(
        "--pocket-max", type=int, default=defaults.pocket_max_structures,
        help="cap structures for pocket detection (0 = all)",
    )
    parser.add_argument(
        "--tier-mode", choices=["percentile", "absolute"], default=defaults.tier_mode,
        help="percentile (cohort-relative) or absolute (fixed score cut-offs)",
    )
    parser.add_argument(
        "--tier1-pct", type=float, default=defaults.tier1_pct,
        help="top fraction -> Tier 1 (percentile mode)",
    )
    parser.add_argument(
        "--tier2-pct", type=float, default=defaults.tier2_pct,
        help="cumulative top fraction -> Tier 2 (percentile mode)",
    )
    parser.add_argument(
        "--tier1-threshold", type=float, default=defaults.tier1_threshold,
        help="minimum score for Tier 1 (absolute mode)",
    )
    parser.add_argument(
        "--tier2-threshold", type=float, default=defaults.tier2_threshold,
        help="minimum score for Tier 2 (absolute mode)",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)

    return Config(
        outdir=args.outdir,
        gram_pos_taxa=tuple(int(x) for x in args.taxa.split(",") if x.strip()),
        deg_file=args.deg_file,
        pocket_enable=not args.no_pocket,
        hard_host_gate=args.hard_host_gate,
        host_identity_cutoff=args.host_identity,
        af_workers=args.af_workers,
        af_refresh=args.refresh_af,
        pocket_max_structures=args.pocket_max,
        tier_mode=args.tier_mode,
        tier1_pct=args.tier1_pct,
        tier2_pct=args.tier2_pct,
        tier1_threshold=args.tier1_threshold,
        tier2_threshold=args.tier2_threshold,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    configure_logging()
    run_pipeline(parse_args(argv))
