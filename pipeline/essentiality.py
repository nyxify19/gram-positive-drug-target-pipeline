"""Gene essentiality scoring from DEG tables or a keyword heuristic."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from pipeline.config import Config, ESSENTIALITY_KEYWORDS, LOGGER


def _essentiality_from_deg(df: pd.DataFrame, deg_path: str) -> Optional[pd.DataFrame]:
    """Score essentiality from a user-supplied DEG-style table, or None."""
    try:
        deg = pd.read_csv(deg_path, sep=None, engine="python")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("[essentiality] could not parse DEG file (%s); using heuristic.", exc)
        return None

    columns = {c.lower(): c for c in deg.columns}
    key = columns.get("accession") or columns.get("gene") or columns.get("gene_name")
    if key is None:
        LOGGER.warning("[essentiality] DEG file lacks accession/gene column; heuristic.")
        return None

    essential = set(deg[key].astype(str).str.strip().str.lower())
    by_accession = df["accession"].str.strip().str.lower().isin(essential)
    by_gene = df["gene_name"].str.strip().str.lower().isin(essential)
    df = df.copy()
    df["essentiality_score"] = (by_accession | by_gene).astype(float)
    df["essentiality_source"] = "DEG"
    LOGGER.info(
        "[essentiality] DEG table matched %d essential proteins.",
        int(df["essentiality_score"].sum()),
    )
    return df


def add_essentiality(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add an essentiality score from a DEG table when present, else a heuristic."""
    deg_path = cfg.path(cfg.deg_file)
    if not os.path.exists(deg_path) and os.path.exists(cfg.deg_file):
        deg_path = cfg.deg_file

    if os.path.exists(deg_path):
        scored = _essentiality_from_deg(df, deg_path)
        if scored is not None:
            return scored

    df = df.copy()
    text = (
        df["function"].fillna("") + " "
        + df["protein_name"].fillna("") + " "
        + df["keywords"].fillna("")
    ).str.lower()
    df["essentiality_score"] = text.str.contains(
        ESSENTIALITY_KEYWORDS, regex=True, na=False
    ).astype(float)
    df["essentiality_source"] = "heuristic"
    LOGGER.info(
        "[essentiality] heuristic flagged %d candidate-essential proteins "
        "(supply --deg-file for curated evidence).",
        int(df["essentiality_score"].sum()),
    )
    return df
