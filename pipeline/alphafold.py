"""AlphaFold pLDDT structural confidence features."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import requests

from pipeline.config import (
    ALPHAFOLD_API_URL, ALPHAFOLD_FILES_URL, Config, LOGGER,
    PLDDT_DISORDERED, PLDDT_HIGH_CONFIDENCE, PLDDT_MAX,
)
from pipeline.utils import cached_parallel_fetch, db_connection, http


def _init_alphafold_cache(db_path: str) -> None:
    """Create the AlphaFold cache table if it does not exist."""
    with db_connection(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS af_results ("
            "accession TEXT PRIMARY KEY, af_mean_plddt REAL, af_frac_high_conf REAL, "
            "af_frac_disordered REAL, af_coverage INTEGER, fetched_at TEXT)"
        )


def _resolve_confidence_url(accession: str, fallback_version: int) -> tuple[Optional[str], Optional[str]]:
    """Resolve the per-residue pLDDT JSON URL for an accession."""
    try:
        response = http().get(ALPHAFOLD_API_URL.format(accession=accession), timeout=20)
        if response.status_code in (400, 404, 422):
            return None, "no_model"
        response.raise_for_status()
        metadata = response.json()
        if not (isinstance(metadata, list) and metadata):
            return None, "empty_api"

        entry = metadata[0]
        for key in ("confidenceUrl", "plddtUrl", "plddtDocUrl"):
            if entry.get(key):
                return entry[key], None

        entry_id = entry.get("entryId")
        version = entry.get("latestVersion") or fallback_version
        if entry_id:
            base = entry_id.rsplit("-model", 1)[0] if "-model" in entry_id else entry_id
            return f"{ALPHAFOLD_FILES_URL}/{base}-confidence_v{version}.json", None
        return None, "empty_api"
    except requests.exceptions.HTTPError as exc:
        return None, f"http_{getattr(exc.response, 'status_code', '?')}"
    except requests.exceptions.RequestException as exc:
        return None, f"net:{type(exc).__name__}"


def _fetch_alphafold_record(accession: str, fallback_version: int) -> tuple:
    """Fetch pLDDT confidence for one accession."""
    url, error = _resolve_confidence_url(accession, fallback_version)
    if error == "no_model":
        return accession, None, None, None, 0, "no_model"
    if url is None:
        return accession, None, None, None, 0, error or "resolve_failed"

    try:
        response = http().get(url, timeout=20)
        if response.status_code == 404:
            legacy = (
                f"{ALPHAFOLD_FILES_URL}/AF-{accession}-F1-"
                f"confidence_v{fallback_version}.json"
            )
            response = http().get(legacy, timeout=20)
        if response.status_code != 200:
            return accession, None, None, None, 0, f"http_{response.status_code}"

        scores = np.asarray(response.json().get("confidenceScore", []), dtype=float)
        if scores.size == 0:
            return accession, None, None, None, 0, "empty_json"
        return (
            accession,
            float(scores.mean()),
            float((scores >= PLDDT_HIGH_CONFIDENCE).mean()),
            float((scores < PLDDT_DISORDERED).mean()),
            1,
            "ok",
        )
    except requests.exceptions.RequestException as exc:
        return accession, None, None, None, 0, f"net:{type(exc).__name__}"


def add_alphafold_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach AlphaFold pLDDT confidence features, using a disk cache."""
    db_path = cfg.path(cfg.af_cache_db)
    _init_alphafold_cache(db_path)

    if cfg.af_refresh:
        with db_connection(db_path) as conn:
            purged = conn.execute("DELETE FROM af_results WHERE af_coverage=0").rowcount
        LOGGER.info("AlphaFold: --refresh-af purged %d cached failures.", purged)

    with db_connection(db_path) as conn:
        cached = {
            row[0]
            for row in conn.execute(
                "SELECT accession FROM af_results WHERE af_coverage=1"
            )
        }
    to_fetch = [acc for acc in df["accession"] if acc not in cached]
    LOGGER.info("AlphaFold: %d cached hits, %d to fetch.", len(cached), len(to_fetch))

    status_counts = cached_parallel_fetch(
        to_fetch,
        cache_db=db_path,
        table="af_results",
        fetch_fn=lambda acc: _fetch_alphafold_record(acc, cfg.af_version),
        workers=cfg.af_workers,
        should_cache=lambda row: row[-1] in ("ok", "no_model"),
        label="AlphaFold",
    )
    if to_fetch:
        _report_fetch_health(status_counts, len(to_fetch))

    with db_connection(db_path) as conn:
        af = pd.read_sql(
            "SELECT accession, af_mean_plddt, af_frac_high_conf, "
            "af_frac_disordered, af_coverage FROM af_results",
            conn,
        )

    df = df.merge(af, on="accession", how="left")
    covered = df["af_coverage"].fillna(0) == 1
    for col in ("af_mean_plddt", "af_frac_high_conf", "af_frac_disordered"):
        median = df.loc[covered, col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(median) else median)
    df["af_coverage"] = df["af_coverage"].fillna(0).astype(int)
    df["af_mean_plddt_norm"] = (df["af_mean_plddt"] / PLDDT_MAX).clip(0, 1)

    covered_n = int(df["af_coverage"].sum())
    LOGGER.info(
        "AlphaFold: %d / %d have predictions (%.1f%%)",
        covered_n, len(df), 100 * covered_n / max(len(df), 1),
    )
    return df


def _report_fetch_health(status_counts: dict[str, int], attempted: int) -> None:
    """Log a diagnostic breakdown so silent total-failures are never hidden."""
    ok = status_counts.get("ok", 0)
    transient = sum(
        n for status, n in status_counts.items() if status not in ("ok", "no_model")
    )
    LOGGER.info("AlphaFold fetch outcomes: %s", status_counts)
    if ok == 0:
        LOGGER.error(
            "[AlphaFold] ZERO successful fetches out of %d attempts. This is "
            "almost always a network problem (firewall / dropped TLS). Failures "
            "were NOT cached; rerun (optionally with --refresh-af) once "
            "connectivity is stable.",
            attempted,
        )
    elif transient > ok:
        LOGGER.warning(
            "[AlphaFold] %d transient failures vs %d successes; coverage is "
            "degraded by network errors. Consider a rerun.",
            transient, ok,
        )
