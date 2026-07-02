"""Constants, weights, and runtime configuration."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger("drugtarget")

# UniProt API
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
HUMAN_PROTEOME_ID = "UP000005640"
UNIPROT_PAGE_SIZE = 250
UNIPROT_FIELDS = (
    "accession,id,gene_names,protein_name,length,sequence,"
    "cc_function,cc_subcellular_location,xref_pdb,xref_drugbank,"
    "organism_id,organism_name,lineage,keyword"
)

# AlphaFold API
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
ALPHAFOLD_FILES_URL = "https://alphafold.ebi.ac.uk/files"
PLDDT_HIGH_CONFIDENCE = 70.0
PLDDT_DISORDERED = 50.0
PLDDT_MAX = 100.0

# Sequence filtering
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MIN_SEQUENCE_LENGTH = 50
MAX_SEQUENCE_LENGTH = 2000

# Pocket proxy
POCKET_PROXY_OPTIMUM_LENGTH = 325.0
POCKET_PROXY_WIDTH = 250.0

# Essentiality keyword pattern
ESSENTIALITY_KEYWORDS = (
    r"\bribosom|\bdna polymerase\b|\brna polymerase\b|\bgyrase\b|"
    r"\btopoisomerase\b|\bcell wall\b|\bpeptidoglycan\b|\baminoacyl-trna\b|"
    r"\belongation factor\b|\bfatty acid\b|\batp synthase\b|"
    r"\bdna replication\b|\bcell division\b|\bseptum\b"
)

# Composite scoring weights
COMPOSITE_WEIGHTS: dict[str, float] = {
    "druggability_proba": 0.30,
    "conservation_score": 0.18,
    "host_selectivity": 0.18,
    "essentiality_score": 0.12,
    "pocket_druggability": 0.12,
    "af_structure_bonus": 0.07,
    "has_function_bonus": 0.03,
}

SCORE_TERMS: dict[str, str] = {
    "druggability_proba": "druggability_proba",
    "conservation_score": "conservation_score",
    "host_selectivity": "host_selectivity",
    "essentiality_score": "essentiality_score",
    "pocket_druggability": "pocket_druggability",
    "af_structure_bonus": "af_mean_plddt_norm",
    "has_function_bonus": "has_function",
}

FEATURE_COLS: list[str] = [
    "length", "molecular_weight_kDa", "pI", "hydropathy_index",
    "aromaticity", "instability_index",
    "is_membrane", "is_cytoplasmic", "has_gene_name",
    "af_mean_plddt", "af_coverage", "pocket_druggability",
]

SCORING_FEATURE_REPORT: list[tuple[str, str, str]] = [
    ("druggability_proba", "ML druggability probability", "check labels/class balance"),
    ("conservation_score", "Cross-strain conservation", "check gene-name coverage"),
    ("host_selectivity", "Human off-target selectivity", "install MMseqs2"),
    ("essentiality_score", "Gene essentiality", "supply --deg-file"),
    ("pocket_druggability", "Binding-pocket druggability", "install p2rank or fpocket"),
    ("af_mean_plddt_norm", "AlphaFold pLDDT confidence", "rerun with --refresh-af"),
    ("has_function", "Functional annotation present", ""),
]

TIER_COLORS = {"Tier 1": "#2ca02c", "Tier 2": "#ff7f0e", "Tier 3": "#9e9e9e"}


@dataclass
class Config:
    """Runtime configuration for a single pipeline run."""

    gram_pos_taxa: tuple[int, ...] = (1239, 201174)

    outdir: str = "drugtarget_out"
    uniprot_cache: str = "grampos_proteome_cache.csv"
    af_cache_db: str = "alphafold_cache.db"
    pocket_cache_db: str = "pocket_cache.db"
    host_fasta: str = "human_proteome_UP000005640.fasta"
    deg_file: str = "deg_essential.tsv"

    af_version: int = 6
    af_workers: int = 8
    af_refresh: bool = False

    host_evalue_cutoff: float = 1e-4
    host_identity_cutoff: float = 35.0
    host_coverage_cutoff: float = 0.25
    mmseqs_sensitivity: float = 5.7

    pocket_enable: bool = True
    pocket_workers: int = 4
    pocket_max_structures: int = 0

    tier_mode: str = "percentile"
    tier1_pct: float = 0.05
    tier2_pct: float = 0.20
    tier1_threshold: float = 0.70
    tier2_threshold: float = 0.50

    mc_n_samples: int = 1000
    mc_dirichlet_a: float = 10.0

    hard_host_gate: bool = False

    seed: int = 42

    def path(self, name: str) -> str:
        """Resolve "name" against the output directory unless it is absolute."""
        return name if os.path.isabs(name) else os.path.join(self.outdir, name)
