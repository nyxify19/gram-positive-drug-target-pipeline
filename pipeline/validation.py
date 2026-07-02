"""Composite-weight validation via Fisher's exact test against known targets."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from pipeline.config import Config, LOGGER


# ── Curated known antibacterial drug targets ─────────────────────────────
# Keys are normalised gene names (lowercase, alphanumeric, trailing digits
# stripped).  Values are (target description, drug class) for reporting.
# Sources: WHO Essential Medicines Model List 2023; Tommasi et al. (2015)
# Nat Rev Drug Discov 14:529; Silver (2011) Clin Microbiol Rev 24:71.
KNOWN_TARGET_GENES: dict[str, tuple[str, str]] = {
    # DNA replication & transcription
    "gyra":  ("DNA gyrase subunit A",                "fluoroquinolones"),
    "gyrb":  ("DNA gyrase subunit B",                "fluoroquinolones"),
    "parc":  ("Topoisomerase IV subunit A",           "fluoroquinolones"),
    "pare":  ("Topoisomerase IV subunit B",           "fluoroquinolones"),
    "rpob":  ("RNA polymerase β subunit",             "rifampicin"),
    "rpoc":  ("RNA polymerase β' subunit",            "fidaxomicin"),
    "dnae":  ("DNA pol III α subunit",                "nargenicin"),
    "dnan":  ("DNA pol III β clamp",                  "griselimycin"),
    # Folate pathway
    "fola":  ("Dihydrofolate reductase",              "trimethoprim"),
    "dfra":  ("Dihydrofolate reductase",              "trimethoprim"),
    "folp":  ("Dihydropteroate synthase",             "sulfonamides"),
    # Cell wall / peptidoglycan
    "ddl":   ("D-Ala–D-Ala ligase",                   "D-cycloserine"),
    "alr":   ("Alanine racemase",                     "D-cycloserine"),
    "mura":  ("MurA transferase",                     "fosfomycin"),
    "murb":  ("MurB reductase",                       "peptidoglycan synthesis"),
    "murc":  ("MurC ligase",                          "peptidoglycan synthesis"),
    "murd":  ("MurD ligase",                          "peptidoglycan synthesis"),
    "mure":  ("MurE ligase",                          "peptidoglycan synthesis"),
    "murf":  ("MurF ligase",                          "peptidoglycan synthesis"),
    "ftsi":  ("PBP 3",                                "β-lactams"),
    "pbp":   ("Penicillin-binding protein",           "β-lactams"),
    "meca":  ("PBP 2a",                               "β-lactams"),
    # Fatty acid synthesis
    "fabi":  ("Enoyl-ACP reductase",                  "triclosan / isoniazid"),
    "fabh":  ("β-ketoacyl-ACP synthase III",          "platencin"),
    "acpp":  ("Acyl carrier protein",                 "fatty acid synthesis"),
    # Translation machinery
    "iles":  ("Isoleucyl-tRNA synthetase",            "mupirocin"),
    "fusa":  ("Elongation factor G",                  "fusidic acid"),
    "tuf":   ("Elongation factor Tu",                 "kirromycin / GE2270A"),
    # Energy metabolism
    "atpe":  ("ATP synthase subunit c",               "bedaquiline"),
    "atpd":  ("ATP synthase subunit β",               "bedaquiline"),
    # Lysine / DAP pathway
    "dapb":  ("Dihydrodipicolinate reductase",        "lysine / DAP pathway"),
    "dapa":  ("Dihydrodipicolinate synthase",         "lysine / DAP pathway"),
}

# Protein-name regex for targets whose gene names are inconsistent across
# strains (common in uncharacterised UniProt entries).
_KNOWN_TARGET_PROTEIN_RE = re.compile(
    r"(?i)"
    r"dna gyrase|topoisomerase iv|dihydrofolate reductase|"
    r"dihydropteroate synthase|d-alanine.{0,3}d-alanine ligase|"
    r"alanine racemase|penicillin.binding protein|"
    r"enoyl.+reductase|isoleucyl.trna synthetase|"
    r"elongation factor g\b|elongation factor tu\b|"
    r"atp synthase subunit [cβb]|rna polymerase.+beta|"
    r"udp-n-acetylglucosamine.*enolpyruvyl"
)


def _normalise_gene(name: str) -> str:
    """Normalise a single gene name (same logic as features._normalise_gene_name)."""
    cleaned = re.sub(r"[^a-z0-9]", "", name.lower())
    return re.sub(r"(?<=[a-z])\d+$", "", cleaned)


def identify_known_targets(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series flagging rows matching curated antibacterial targets."""
    norm_genes = df["gene_name"].fillna("").apply(_normalise_gene)
    by_gene = norm_genes.isin(KNOWN_TARGET_GENES)
    by_protein = df["protein_name"].fillna("").str.contains(
        _KNOWN_TARGET_PROTEIN_RE, na=False,
    )
    return by_gene | by_protein


def validate_tier_enrichment(df: pd.DataFrame, cfg: Config) -> dict:
    """Fisher's exact test: are known antibacterial targets enriched in Tier 1?

    Builds a 2×2 contingency table:

    ========================  =========  ===========
                               Tier 1     Not Tier 1
    ========================  =========  ===========
    Known target               a          b
    Other protein              c          d
    ========================  =========  ===========

    Returns a results dict suitable for inclusion in the run manifest.
    """
    if "priority_tier" not in df.columns:
        LOGGER.warning("[validation] no priority_tier column; skipping enrichment test.")
        return {}

    is_known = identify_known_targets(df)
    n_known = int(is_known.sum())
    if n_known == 0:
        LOGGER.warning(
            "[validation] no known antibacterial targets matched in the dataset; "
            "enrichment test skipped.  This may happen if the organism set has "
            "very different gene nomenclature."
        )
        return {"known_targets_matched": 0, "note": "no matches found"}

    in_tier1 = df["priority_tier"] == "Tier 1"

    # 2×2 contingency table
    a = int((is_known & in_tier1).sum())       # known AND Tier 1
    b = int((is_known & ~in_tier1).sum())      # known AND NOT Tier 1
    c = int((~is_known & in_tier1).sum())      # NOT known AND Tier 1
    d = int((~is_known & ~in_tier1).sum())     # NOT known AND NOT Tier 1

    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    tier1_rate_known = a / max(a + b, 1)
    tier1_rate_other = c / max(c + d, 1)
    fold_enrichment = tier1_rate_known / max(tier1_rate_other, 1e-9)

    # ── Detailed log ──
    matched = df.loc[is_known, ["accession", "gene_name", "protein_name", "priority_tier"]]
    matched_tier1 = matched[matched["priority_tier"] == "Tier 1"]

    LOGGER.info("=" * 70)
    LOGGER.info("[validation] COMPOSITE-WEIGHT VALIDATION — Fisher's exact test")
    LOGGER.info("-" * 70)
    LOGGER.info(
        "Known antibacterial targets matched: %d / %d proteome proteins "
        "(%d curated gene names checked)",
        n_known, len(df), len(KNOWN_TARGET_GENES),
    )
    LOGGER.info(
        "Contingency table:\n"
        "                   Tier 1    Other\n"
        "  Known target    %5d     %5d\n"
        "  Other protein   %5d     %5d",
        a, b, c, d,
    )
    LOGGER.info(
        "Tier 1 rate: known targets = %.1f%%  |  background = %.1f%%  "
        "(%.1f× enrichment)",
        100 * tier1_rate_known, 100 * tier1_rate_other, fold_enrichment,
    )
    LOGGER.info(
        "Fisher's exact test (one-sided): odds ratio = %.2f, p = %.2e",
        odds_ratio, p_value,
    )
    if p_value < 0.05:
        LOGGER.info(
            "✓ Tier 1 is significantly enriched for known targets (p < 0.05).  "
            "Composite weights produce biologically meaningful rankings."
        )
    else:
        LOGGER.warning(
            "✗ Enrichment not significant at α=0.05 (p = %.3f).  Consider "
            "reviewing composite weights or tier thresholds.", p_value,
        )
    if not matched_tier1.empty:
        LOGGER.info(
            "Known targets recovered in Tier 1:\n%s",
            matched_tier1[["accession", "gene_name", "protein_name"]]
            .to_string(index=False),
        )
    LOGGER.info("=" * 70)

    return {
        "known_targets_matched": n_known,
        "known_in_tier1": a,
        "known_not_tier1": b,
        "other_in_tier1": c,
        "other_not_tier1": d,
        "tier1_rate_known_pct": round(100 * tier1_rate_known, 1),
        "tier1_rate_background_pct": round(100 * tier1_rate_other, 1),
        "fold_enrichment": round(float(fold_enrichment), 2),
        "odds_ratio": round(float(odds_ratio), 2) if np.isfinite(odds_ratio) else "inf",
        "p_value": float(p_value),
        "significant_at_005": bool(p_value < 0.05),
    }
