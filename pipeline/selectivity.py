"""Host (human) off-target selectivity via subtractive genomics."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import pandas as pd
import requests

from pipeline.config import (
    COMPOSITE_WEIGHTS, Config, HUMAN_PROTEOME_ID, LOGGER, UNIPROT_STREAM_URL,
)
from pipeline.utils import find_tool, http


def _download_host_proteome(cfg: Config) -> Optional[str]:
    """Download the human reference proteome FASTA once; return its path."""
    path = cfg.path(cfg.host_fasta)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = (
        f"{UNIPROT_STREAM_URL}?query=proteome:{HUMAN_PROTEOME_ID}"
        "&format=fasta&compressed=false"
    )
    try:
        LOGGER.info("Downloading human reference proteome (%s)...", HUMAN_PROTEOME_ID)
        with http().get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        LOGGER.info("Human proteome saved (%.1f MB).", os.path.getsize(path) / 1e6)
        return path
    except requests.exceptions.RequestException as exc:
        LOGGER.warning("Could not download host proteome: %s", exc)
        return None


def _run_host_homology_search(
    df: pd.DataFrame, mmseqs_bin: str, host_fasta: str, cfg: Config
) -> Optional[pd.DataFrame]:
    """Run MMseqs2 of candidates against the host proteome."""
    with tempfile.TemporaryDirectory() as tmp:
        query_fasta = os.path.join(tmp, "candidates.fasta")
        with open(query_fasta, "w") as handle:
            for accession, sequence in zip(df["accession"], df["sequence"]):
                handle.write(f">{accession}\n{sequence}\n")

        result_file = os.path.join(tmp, "host_hits.m8")
        command = [
            mmseqs_bin, "easy-search", query_fasta, host_fasta, result_file,
            os.path.join(tmp, "mmtmp"),
            "-a", "-s", str(cfg.mmseqs_sensitivity),
            "--format-output", "query,target,pident,alnlen,evalue,bits,qcov",
            "--threads", str(os.cpu_count() or 4),
        ]
        LOGGER.info("[host] running MMseqs2 vs human proteome (%d candidates)...", len(df))
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            tail = exc.stderr.strip().splitlines()[-1] if exc.stderr else exc
            LOGGER.warning("[host] mmseqs failed (%s); skipping host filter.", tail)
            return None

        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            return pd.DataFrame(columns=["query"])

        hits = pd.read_csv(
            result_file, sep="\t", header=None,
            names=["query", "target", "pident", "alnlen", "evalue", "bits", "qcov"],
        )
        return hits.sort_values("evalue").groupby("query", as_index=False).first()


def add_host_selectivity(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add human off-target selectivity via subtractive genomics (MMseqs2)."""
    df = df.copy()
    df["human_best_identity"] = 0.0
    df["human_best_evalue"] = 10.0
    df["human_qcov"] = 0.0
    df["is_host_homologous"] = 0
    df["host_selectivity"] = 0.5
    df["host_selectivity_source"] = "placeholder"

    mmseqs_bin = find_tool("mmseqs", "mmseqs.bat")
    if not mmseqs_bin:
        weight = 100 * COMPOSITE_WEIGHTS["host_selectivity"]
        LOGGER.warning("=" * 70)
        LOGGER.warning(
            "[host] MMseqs2 not found; host selectivity flat-lined at 0.5. "
            "Install: conda install -c bioconda mmseqs2"
        )
        LOGGER.warning("=" * 70)
        return df

    host_fasta = _download_host_proteome(cfg)
    if not host_fasta:
        LOGGER.warning("[host] no host proteome; skipping host filter.")
        return df

    best_hits = _run_host_homology_search(df, mmseqs_bin, host_fasta, cfg)
    if best_hits is None:
        return df
    df["host_selectivity_source"] = "mmseqs2"
    if best_hits.empty:
        LOGGER.info("[host] no human hits found (fully selective candidate set).")
        df["host_selectivity"] = 1.0
        return df

    best = best_hits.drop_duplicates("query").set_index("query")
    df["human_best_identity"] = df["accession"].map(best["pident"]).fillna(0.0)
    df["human_best_evalue"] = df["accession"].map(best["evalue"]).fillna(10.0)
    df["human_qcov"] = df["accession"].map(best["qcov"]).fillna(0.0)

    is_homologous = (
        (df["human_best_evalue"] <= cfg.host_evalue_cutoff)
        & (df["human_best_identity"] >= cfg.host_identity_cutoff)
        & (df["human_qcov"] >= cfg.host_coverage_cutoff)
    )
    df["is_host_homologous"] = is_homologous.astype(int)
    df["host_selectivity"] = (1.0 - df["human_best_identity"] / 100.0).clip(0, 1)

    LOGGER.info(
        "[host] %d / %d candidates are human-homologous (off-target risk).",
        int(df["is_host_homologous"].sum()), len(df),
    )
    return df
