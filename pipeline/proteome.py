"""UniProt proteome retrieval, parsing, and cleaning."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from pipeline.config import (
    Config, LOGGER, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS, UNIPROT_FIELDS, UNIPROT_PAGE_SIZE, UNIPROT_SEARCH_URL,
)
from pipeline.utils import http


def _build_taxon_query(taxa: Iterable[int]) -> str:
    """Return a reviewed-only UniProtKB query for the given taxonomy IDs."""
    clause = " OR ".join(f"taxonomy_id:{t}" for t in taxa)
    return f"({clause}) AND reviewed:true"


def _parse_uniprot_entry(entry: dict) -> Optional[dict]:
    """Flatten one UniProtKB JSON entry to a record, or None if unusable."""
    accession = entry.get("primaryAccession", "").strip()
    sequence = entry.get("sequence", {}).get("value", "").strip()
    if not accession or not sequence:
        return None

    description = entry.get("proteinDescription", {})
    protein_name = (
        description.get("recommendedName", {}).get("fullName", {}).get("value", "")
    )
    genes = entry.get("genes", [])
    gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""

    function, location = "", ""
    for comment in entry.get("comments", []):
        comment_type = comment.get("commentType", "").upper().replace("_", " ")
        if comment_type == "FUNCTION":
            texts = comment.get("texts", [])
            function = texts[0].get("value", "") if texts else function
        elif comment_type == "SUBCELLULAR LOCATION":
            locations = comment.get("subcellularLocations", [])
            if locations:
                location = locations[0].get("location", {}).get("value", "")

    pdb_ids, drugbank_ids = [], []
    for xref in entry.get("uniProtKBCrossReferences", []):
        database = xref.get("database", "")
        if database == "PDB":
            pdb_ids.append(xref.get("id", ""))
        elif database == "DrugBank":
            drugbank_ids.append(xref.get("id", ""))

    keywords = [kw.get("name", "") for kw in entry.get("keywords", []) if kw.get("name")]

    return {
        "accession": accession,
        "entry_name": entry.get("uniProtkbId", ""),
        "protein_name": protein_name,
        "gene_name": gene_name,
        "length": entry.get("sequence", {}).get("length", 0),
        "sequence": sequence,
        "function": function,
        "subcellular_location": location,
        "pdb_structures": "; ".join(p for p in pdb_ids if p),
        "drugbank_targets": "; ".join(d for d in drugbank_ids if d),
        "keywords": "; ".join(keywords),
        "organism_id": entry.get("organism", {}).get("taxonId", 0),
        "organism_name": entry.get("organism", {}).get("scientificName", ""),
    }


def parse_uniprot_entries(entries: Iterable[dict]) -> pd.DataFrame:
    """Convert raw UniProtKB JSON entries into a tidy DataFrame."""
    records, skipped = [], 0
    for entry in entries:
        record = _parse_uniprot_entry(entry)
        if record is None:
            skipped += 1
        else:
            records.append(record)
    if skipped:
        LOGGER.warning("Skipped %d entries (missing accession/sequence).", skipped)
    LOGGER.info("Processed %d proteins.", len(records))
    return pd.DataFrame(records)


def _load_proteome_cache(cache: str, taxa: tuple[int, ...]) -> Optional[pd.DataFrame]:
    """Return the cached proteome if present and built from the same taxa."""
    if not os.path.exists(cache):
        return None

    meta_path = cache + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
        if sorted(meta.get("taxa", [])) != sorted(taxa):
            LOGGER.warning(
                "[cache] taxa changed (cached %s vs requested %s); ignoring cache.",
                meta.get("taxa"), list(taxa),
            )
            return None
        release = meta.get("uniprot_release", "unknown")
    else:
        LOGGER.warning("[cache] no metadata sidecar; cannot verify taxa/release.")
        release = "unknown"

    LOGGER.info("Loading proteome from cache: %s (release %s)", cache, release)
    df = clean_proteome(pd.read_csv(cache, low_memory=False))
    df.attrs["uniprot_release"] = release
    return df


def fetch_gram_positive_proteome(cfg: Config) -> pd.DataFrame:
    """Fetch (or load from cache) the reviewed gram-positive proteome."""
    cache = cfg.path(cfg.uniprot_cache)
    cached = _load_proteome_cache(cache, cfg.gram_pos_taxa)
    if cached is not None:
        return cached

    params: Optional[dict] = {
        "query": _build_taxon_query(cfg.gram_pos_taxa),
        "format": "json",
        "size": UNIPROT_PAGE_SIZE,
        "fields": UNIPROT_FIELDS,
    }
    url: Optional[str] = UNIPROT_SEARCH_URL
    entries: list[dict] = []
    release: Optional[str] = None

    LOGGER.info("Fetching gram-positive proteome from UniProt...")
    try:
        while url:
            response = http().get(url, params=params, timeout=60)
            response.raise_for_status()
            release = release or response.headers.get("X-UniProt-Release", "unknown")
            entries.extend(response.json().get("results", []))
            LOGGER.info("  Fetched %d proteins so far...", len(entries))
            next_match = re.search(
                r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", "")
            )
            url = next_match.group(1) if next_match else None
            params = None
            if url:
                time.sleep(0.2)
    except requests.exceptions.RequestException as exc:
        LOGGER.error("Could not reach UniProt after retries: %s", exc)
        LOGGER.error(
            "This is almost always a network/firewall/proxy block, not a code "
            "error. Try: (1) a different network or hotspot; (2) confirm access "
            "in a browser: %s?query=reviewed:true&size=1&format=json ; "
            "(3) set HTTPS_PROXY if behind a proxy.",
            UNIPROT_SEARCH_URL,
        )
        raise SystemExit(1) from exc

    LOGGER.info("Total fetched: %d proteins (UniProt release %s).", len(entries), release)
    df = parse_uniprot_entries(entries)
    df.attrs["uniprot_release"] = release
    df.to_csv(cache, index=False)
    with open(cache + ".meta.json", "w") as handle:
        json.dump(
            {"taxa": list(cfg.gram_pos_taxa), "uniprot_release": release}, handle
        )
    LOGGER.info("Proteome cached to %s", cache)
    return df


def clean_proteome(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise sequences and drop unusable rows."""
    df = df[df["sequence"].notna() & (df["sequence"] != "")].copy()
    df["sequence"] = (
        df["sequence"].astype(str).str.upper()
        .str.replace(f"[^{STANDARD_AMINO_ACIDS}]", "", regex=True)
    )
    df = df[df["sequence"].str.len() > 0].copy()
    df["length"] = df["sequence"].str.len()
    df = df[
        (df["length"] >= MIN_SEQUENCE_LENGTH) & (df["length"] <= MAX_SEQUENCE_LENGTH)
    ].copy()

    text_cols = [
        "function", "subcellular_location", "gene_name", "protein_name",
        "keywords", "organism_name",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("")
    df["organism_id"] = df["organism_id"].fillna(0).astype(int)

    df = df.drop_duplicates(subset="accession").reset_index(drop=True)
    LOGGER.info(
        "After cleaning: %d proteins across %d organisms.",
        len(df), df["organism_id"].nunique(),
    )
    return df
