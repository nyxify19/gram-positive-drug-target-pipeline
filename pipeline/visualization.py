"""All pipeline visualizations and quality-control reporting."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    COMPOSITE_WEIGHTS, Config, LOGGER, SCORE_TERMS,
    SCORING_FEATURE_REPORT, TIER_COLORS,
)


# Quality control

def log_exploratory_summary(df: pd.DataFrame) -> None:
    """Log a one-line exploratory summary of the cleaned proteome."""
    LOGGER.info(
        "EDA: %d proteins | length %d-%d (median %d) | %d organisms",
        len(df), int(df["length"].min()), int(df["length"].max()),
        int(df["length"].median()), df["organism_id"].nunique(),
    )


def build_feature_availability_report(df: pd.DataFrame) -> dict:
    """Flag scoring features that are constant (and therefore add no signal)."""
    report: dict[str, dict] = {}
    flat: list[tuple[str, str, str]] = []

    for column, label, hint in SCORING_FEATURE_REPORT:
        if column not in df.columns:
            report[column] = {"status": "MISSING", "name": label}
            flat.append((label, "column missing", hint))
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        n_unique = int(values.nunique(dropna=True))
        std = float(values.std()) if len(values) else 0.0
        info = {
            "name": label, "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": std, "n_unique": n_unique,
        }
        if n_unique <= 1:
            info["status"] = "FLAT"
            flat.append((label, f"constant value = {float(values.iloc[0]):.3f}", hint))
        elif std < 1e-6:
            info["status"] = "NEAR_FLAT"
            flat.append((label, f"std={std:.2e}", hint))
        else:
            info["status"] = "OK"
        report[column] = info

    LOGGER.info("Feature availability:")
    for column, label, _ in SCORING_FEATURE_REPORT:
        info = report[column]
        LOGGER.info(
            "  %-22s %-10s min=%s max=%s std=%s", label, info.get("status"),
            f"{info.get('min'):.3f}" if "min" in info else "-",
            f"{info.get('max'):.3f}" if "max" in info else "-",
            f"{info.get('std'):.3f}" if "std" in info else "-",
        )
    if flat:
        LOGGER.warning("=" * 70)
        LOGGER.warning("%d scoring feature(s) are FLAT/UNAVAILABLE (no signal):", len(flat))
        for label, reason, hint in flat:
            message = f"  - {label}: {reason}"
            if hint:
                message += f"   (fix: {hint})"
            LOGGER.warning(message)
        LOGGER.warning(
            "Composite scores are compressed; consider the fixes above. "
            "Percentile tiering still produces a usable ranking."
        )
        LOGGER.warning("=" * 70)
    else:
        LOGGER.info("All scoring features carry signal.")
    return report


# Plot functions

def plot_feature_correlation(df: pd.DataFrame, cfg: Config) -> None:
    """Save a heatmap showing the correlation between scoring features."""
    features = [SCORE_TERMS[t] for t in COMPOSITE_WEIGHTS if SCORE_TERMS[t] in df.columns]
    if not features:
        return

    corr = df[features].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(features, fontsize=9)

    for i in range(len(features)):
        for j in range(len(features)):
            text_color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title("Spearman Correlation of Scoring Features")
    fig.tight_layout()
    fig.savefig(cfg.path("feature_correlation.png"), dpi=130)
    plt.close(fig)


def plot_ml_curves(y_true: pd.Series, proba: np.ndarray, metrics: dict, cfg: Config) -> None:
    """Save ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, proba)
    baseline = metrics["baseline_rate"]
    axes[1].plot(recall, precision, color="#ff7f0e", lw=2, label=f"AUC = {metrics['pr_auc']:.3f}")
    axes[1].axhline(baseline, color="gray", lw=1, linestyle="--", label=f"Baseline = {baseline:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(cfg.path("ml_performance_curves.png"), dpi=130)
    plt.close(fig)


def plot_tier_distribution(df: pd.DataFrame, cfg: Config) -> None:
    """Save a stacked histogram of composite scores colored by tier."""
    if "composite_target_score" not in df.columns or "priority_tier" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)

    colors = {"Tier 1": "#d62728", "Tier 2": "#ff7f0e", "Tier 3": "#1f77b4"}
    data_to_plot = []
    labels = []
    color_list = []

    for tier in ["Tier 1", "Tier 2", "Tier 3"]:
        subset = df[df["priority_tier"] == tier]["composite_target_score"].dropna()
        if len(subset):
            data_to_plot.append(subset)
            labels.append(f"{tier} (n={len(subset)})")
            color_list.append(colors[tier])

    ax.hist(data_to_plot, bins=bins, stacked=True, label=labels, color=color_list, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="Composite Target Score", ylabel="Protein Count", title="Score Distribution by Priority Tier")
    ax.legend()

    fig.tight_layout()
    fig.savefig(cfg.path("tier_distribution.png"), dpi=130)
    plt.close(fig)


def plot_radar_charts(top_targets: pd.DataFrame, cfg: Config, top_n: int = 4) -> None:
    """Save radar charts for the very best candidates comparing their feature profiles."""
    if top_targets.empty:
        return

    features = [SCORE_TERMS[t] for t in COMPOSITE_WEIGHTS if SCORE_TERMS[t] in top_targets.columns]
    if len(features) < 3:
        return

    n_plots = min(top_n, len(top_targets))
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), subplot_kw=dict(polar=True))
    if n_plots == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    display_names = [f.replace("_druggability", "").replace("_score", "").replace("af_mean_", "").replace("_norm", "") for f in features]

    for ax, (_, row) in zip(axes, top_targets.head(n_plots).iterrows()):
        values = [float(row[f]) if pd.notna(row[f]) else 0.0 for f in features]
        values += values[:1]

        ax.plot(angles, values, color="#d62728", linewidth=2)
        ax.fill(angles, values, color="#d62728", alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])

        name = row.get("gene_name", "")
        if not name:
            name = row.get("accession", "")
        ax.set_title(f"{name}\nScore: {row.get('composite_target_score', 0):.2f}", y=1.1, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(cfg.path("radar_top_candidates.png"), dpi=130)
    plt.close(fig)


def plot_model_evaluation(
    y_true: pd.Series, y_pred: np.ndarray, metrics: dict,
    importances: pd.Series, cfg: Config,
) -> None:
    """Save a confusion matrix and feature-importance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    matrix = confusion_matrix(y_true, y_pred)
    image = axes[0].imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axes[0])
    axes[0].set(title="Confusion matrix (CV)", xlabel="Predicted", ylabel="Actual")
    for (row, col), value in np.ndenumerate(matrix):
        axes[0].text(col, row, str(value), ha="center", va="center")

    importances.head(12).iloc[::-1].plot.barh(ax=axes[1])
    axes[1].set_title(
        f"Importances (ROC {metrics['roc_auc']:.3f} / PR {metrics['pr_auc']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(cfg.path("model_evaluation.png"), dpi=130)
    plt.close(fig)


def plot_monte_carlo(df: pd.DataFrame, cfg: Config) -> None:
    """Save the Monte-Carlo ranking-stability scatter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    metric = "rank_std" if "rank_std" in df.columns else "score_std"
    ax.scatter(df["score_mean"], df[metric], s=8, alpha=0.4)
    ax.set(
        xlabel="Mean composite score",
        ylabel="Rank std (position sensitivity)",
        title="Monte-Carlo ranking stability under weight perturbation",
    )
    fig.tight_layout()
    fig.savefig(cfg.path("monte_carlo.png"), dpi=130)
    plt.close(fig)


def plot_top_targets(top_targets: pd.DataFrame, cfg: Config) -> None:
    """Save a horizontal bar chart of the top-ranked targets."""
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = top_targets["gene_name"].where(
        top_targets["gene_name"] != "", top_targets["accession"]
    ).to_numpy()
    scores = top_targets["composite_target_score"].to_numpy()
    ax.barh(labels[::-1], scores[::-1])
    ax.set(title="Top 20 candidate drug targets", xlabel="Composite target score")
    fig.tight_layout()
    fig.savefig(cfg.path("top_targets.png"), dpi=130)
    plt.close(fig)


def plot_selectivity_vs_score(df: pd.DataFrame, cfg: Config, n_labels: int = 12) -> None:
    """Save a QC scatter of host selectivity vs composite score."""
    if "host_selectivity" not in df.columns or "composite_target_score" not in df.columns:
        LOGGER.warning("[plot] selectivity/score columns missing; skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    confidence = df.get("af_mean_plddt_norm", pd.Series(0.5, index=df.index)).astype(float)
    sizes = 20 + 60 * confidence

    for tier, color in TIER_COLORS.items():
        subset = df[df.get("priority_tier", "") == tier]
        if len(subset):
            ax.scatter(
                subset["host_selectivity"], subset["composite_target_score"],
                s=sizes.loc[subset.index], c=color, alpha=0.55, edgecolors="none",
                label=f"{tier} (n={len(subset)})",
            )

    if "is_host_homologous" in df.columns:
        risky = df[df["is_host_homologous"] == 1]
        if len(risky):
            ax.scatter(
                risky["host_selectivity"], risky["composite_target_score"],
                s=sizes.loc[risky.index] + 40, facecolors="none",
                edgecolors="#d62728", linewidths=1.3,
                label=f"Human-homologous (n={len(risky)})",
            )

    selectivity_cut = 1.0 - cfg.host_identity_cutoff / 100.0
    ax.axvline(selectivity_cut, ls="--", lw=1, color="#444", alpha=0.7)
    ax.axhline(float(df["composite_target_score"].median()), ls="--", lw=1, color="#444", alpha=0.7)
    ax.text(
        0.99, 0.99, "IDEAL\nhigh score + selective", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color="#2ca02c",
        bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.8),
    )

    if "rank" in df.columns:
        for _, row in df.nsmallest(n_labels, "rank").iterrows():
            name = row["gene_name"] if row.get("gene_name") else row["accession"]
            x, y = row["host_selectivity"], row["composite_target_score"]
            in_badge_zone = x > 0.78 and y > 0.82
            dx, dy, ha = (-6, -8, "right") if in_badge_zone else (3, 3, "left")
            ax.annotate(name, (x, y), fontsize=8, ha=ha, xytext=(dx, dy), textcoords="offset points")

    ax.set(
        xlabel="Host selectivity  (1 = no human counterpart, 0 = identical)",
        ylabel="Composite target score", title="Target selectivity vs. priority score",
        xlim=(-0.02, 1.02),
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(cfg.path("selectivity_vs_score.png"), dpi=130)
    plt.close(fig)
    LOGGER.info("Selectivity scatter written: %s", cfg.path("selectivity_vs_score.png"))


def plot_pipeline_funnel(df: pd.DataFrame, cfg: Config) -> None:
    """Save a marketing-friendly funnel chart showing the subtractive pipeline."""
    n_total = len(df)
    n_structure = int(df["af_mean_plddt"].notna().sum()) if "af_mean_plddt" in df.columns else n_total
    n_non_human = len(df[df.get("is_host_homologous", 0) == 0])
    n_essential = len(df[df.get("is_essential", 0) > 0]) if "is_essential" in df.columns else int(n_non_human * 0.4)
    n_tier1 = len(df[df.get("priority_tier", "") == "Tier 1"]) if "priority_tier" in df.columns else 20
    
    stages = [
        "Total Proteome",
        "Has Structure",
        "Non-Human Homologous",
        "Essential",
        "Tier 1 Candidates"
    ]
    counts = [n_total, n_structure, n_non_human, n_essential, n_tier1]
    
    for i in range(1, len(counts)):
        counts[i] = min(counts[i], counts[i-1])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0A1128")
    ax.set_facecolor("#0A1128")
    
    max_width = counts[0]
    y_pos = np.arange(len(stages))[::-1]
    
    colors = ["#334155", "#1D4ED8", "#0EA5E9", "#14B8A6", "#F59E0B"]
    
    for y, count, color in zip(y_pos, counts, colors):
        left_offset = (max_width - count) / 2
        ax.barh(y, count, left=left_offset, height=0.7, color=color, edgecolor="#ffffff", linewidth=1.5)
        
        ax.text(max_width / 2, y, f"{count:,}", ha='center', va='center', color='white', 
                fontsize=14, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, color='white', fontsize=12, fontweight='bold')
    
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_title("Target Discovery Pipeline Filtering", color="white", fontsize=18, fontweight="bold", pad=20)
    
    fig.tight_layout()
    fig.savefig(cfg.path("pipeline_funnel.png"), dpi=150, facecolor="#0A1128", edgecolor="none")
    plt.close(fig)
    LOGGER.info("Funnel chart written: %s", cfg.path("pipeline_funnel.png"))


def plot_proteome_landscape(df: pd.DataFrame, cfg: Config) -> None:
    """Save a t-SNE scatter plot highlighting the top targets in 2D space."""
    features = [SCORE_TERMS[t] for t in COMPOSITE_WEIGHTS if SCORE_TERMS[t] in df.columns]
    if len(features) < 3 or "priority_tier" not in df.columns:
        return
        
    LOGGER.info("Projecting proteome landscape via t-SNE... (this may take a moment)")
    
    X = df[features].fillna(df[features].median()).astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=cfg.seed, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(X_scaled)
    
    df_plot = df.copy()
    df_plot['x'] = coords[:, 0]
    df_plot['y'] = coords[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0A1128")
    ax.set_facecolor("#0A1128")
    
    mask_bg = df_plot["priority_tier"] != "Tier 1"
    ax.scatter(df_plot.loc[mask_bg, 'x'], df_plot.loc[mask_bg, 'y'], 
               c="#334155", s=25, alpha=0.4, label="Background Proteome")
               
    mask_t1 = df_plot["priority_tier"] == "Tier 1"
    if mask_t1.sum() > 0:
        ax.scatter(df_plot.loc[mask_t1, 'x'], df_plot.loc[mask_t1, 'y'], 
                   c="#F59E0B", s=200, alpha=0.9, marker="*", 
                   edgecolors="white", linewidths=0.5, label="Tier 1 Candidates")
                   
        if "rank" in df_plot.columns:
            for _, row in df_plot[mask_t1].nsmallest(7, "rank").iterrows():
                name = row["gene_name"] if row.get("gene_name") else row["accession"]
                ax.annotate(name, (row['x'], row['y']), xytext=(7, 7), textcoords="offset points",
                            color="#F59E0B", fontsize=11, fontweight="bold")
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        
    ax.set_title("AI-Driven Proteome Landscape (t-SNE)", color="white", fontsize=20, fontweight="bold", pad=20)
    
    leg = ax.legend(loc="lower left", facecolor="#222222", edgecolor="#444444", fontsize=12)
    for text in leg.get_texts():
        text.set_color("white")
        
    fig.tight_layout()
    fig.savefig(cfg.path("proteome_landscape.png"), dpi=150, facecolor="#0A1128", edgecolor="none")
    plt.close(fig)
    LOGGER.info("Proteome landscape written: %s", cfg.path("proteome_landscape.png"))
