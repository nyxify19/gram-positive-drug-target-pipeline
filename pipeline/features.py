"""Sequence-derived features: annotations, physicochemistry, conservation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from pipeline.config import LOGGER


def add_annotation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags derived from UniProt annotations."""
    df = df.copy()
    location = df["subcellular_location"].str.lower()
    df["is_membrane"] = location.str.contains("membrane", na=False).astype(int)
    df["is_cytoplasmic"] = location.str.contains("cytoplasm", na=False).astype(int)
    df["has_gene_name"] = (df["gene_name"].str.len() > 0).astype(int)
    df["has_function"] = (df["function"].str.len() > 0).astype(int)
    df["has_structure"] = (df["pdb_structures"].str.len() > 0).astype(int)
    return df


def add_physicochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Biopython physicochemical descriptors per sequence."""
    df = df.copy()
    columns: dict[str, list[float]] = {
        "molecular_weight_kDa": [], "pI": [], "hydropathy_index": [],
        "aromaticity": [], "instability_index": [],
    }
    for sequence in df["sequence"]:
        try:
            analysis = ProteinAnalysis(str(sequence))
            columns["molecular_weight_kDa"].append(analysis.molecular_weight() / 1000.0)
            columns["pI"].append(analysis.isoelectric_point())
            columns["hydropathy_index"].append(analysis.gravy())
            columns["aromaticity"].append(analysis.aromaticity())
            columns["instability_index"].append(analysis.instability_index())
        except (KeyError, ValueError):
            for values in columns.values():
                values.append(np.nan)
    for name, values in columns.items():
        df[name] = pd.Series(values, index=df.index).fillna(np.nanmedian(values))
    LOGGER.info("Physicochemical features computed for %d proteins.", len(df))
    return df


def _normalise_gene_name(names: pd.Series) -> pd.Series:
    """Collapse spelling/case variants so orthologues group together."""
    cleaned = (
        names.fillna("").astype(str).str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    return cleaned.str.replace(r"(?<=[a-z])\d+$", "", regex=True)


def _conservation_key(df: pd.DataFrame) -> pd.Series:
    """Build a cross-species grouping key for each protein."""
    gene = _normalise_gene_name(df["gene_name"])
    protein = (
        df["protein_name"].fillna("").astype(str).str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    )
    key = gene.radd("gene:")
    key = key.mask(gene == "", protein.radd("prot:"))
    blank = key.isin(["gene:", "prot:"])
    return key.mask(blank, "uniq:" + df["accession"].astype(str))


def add_conservation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score conservation as the fraction of organisms sharing a protein's key."""
    df = df.copy()
    total_organisms = max(df["organism_id"].nunique(), 1)
    df["_conservation_key"] = _conservation_key(df)
    organism_counts = (
        df.groupby("_conservation_key")["organism_id"].nunique().rename("_count")
    )
    df = df.join(organism_counts, on="_conservation_key")
    df["conservation_score"] = (df["_count"] / total_organisms).clip(0, 1)
    median = df["conservation_score"].median()
    df["conservation_score"] = df["conservation_score"].fillna(
        0.0 if pd.isna(median) else median
    )
    df = df.drop(columns=["_count", "_conservation_key"])

    LOGGER.info(
        "Conservation computed (normalised gene/protein-name proxy). "
        "Median %.3f, max %.3f",
        df["conservation_score"].median(), df["conservation_score"].max(),
    )
    return df
