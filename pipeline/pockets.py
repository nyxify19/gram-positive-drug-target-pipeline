"""Binding-pocket druggability via P2Rank, fpocket, or a length proxy."""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.config import (
    ALPHAFOLD_FILES_URL, Config, LOGGER,
    POCKET_PROXY_OPTIMUM_LENGTH, POCKET_PROXY_WIDTH,
)
from pipeline.utils import cached_parallel_fetch, db_connection, find_tool, http


def _init_pocket_cache(db_path: str) -> None:
    """Create the pocket cache table if it does not exist."""
    with db_connection(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pockets ("
            "accession TEXT PRIMARY KEY, pocket_score REAL, n_pockets INTEGER, "
            "tool TEXT, fetched_at TEXT)"
        )


def _auto_detect_java_home() -> Optional[str]:
    """Try to find a usable JAVA_HOME on Windows; return path or None."""
    if sys.platform != "win32":
        return None
    for pattern in (
        r"C:\Program Files\Java\*",
        r"C:\Program Files (x86)\Java\*",
    ):
        for d in sorted(glob.glob(pattern), reverse=True):
            if os.path.isfile(os.path.join(d, "bin", "java.exe")):
                return d
    return None


def _download_alphafold_model(accession: str, version: int, dest: str) -> bool:
    """Download an AlphaFold PDB model to ``dest``; return success."""
    versions = [version, *(v for v in (6, 5, 4, 3) if v != version)]
    for candidate in versions:
        url = f"{ALPHAFOLD_FILES_URL}/AF-{accession}-F1-model_v{candidate}.pdb"
        response = http().get(url, timeout=30)
        if response.status_code == 200 and response.content:
            with open(dest, "wb") as handle:
                handle.write(response.content)
            return True
    return False


def _run_p2rank(
    pdb_path: str, workdir: str, *, prank_exe: str, java_home: Optional[str],
) -> Optional[tuple[float, int]]:
    """Run P2Rank and return (top pocket probability, pocket count)."""
    out_dir = os.path.join(workdir, "p2rank_out")

    # Build a clean environment with JAVA_HOME + limited heap
    env = os.environ.copy()
    env["JAVA_OPTS"] = "-Xmx1G"
    if java_home:
        env["JAVA_HOME"] = java_home

    subprocess.run(
        [prank_exe, "predict", "-f", pdb_path, "-o", out_dir,
         "-threads", "1", "-visualizations", "0"],
        check=True, capture_output=True, text=True, env=env,
    )
    predictions = os.path.join(out_dir, os.path.basename(pdb_path) + "_predictions.csv")
    if not os.path.exists(predictions):
        return None
    table = pd.read_csv(predictions, skipinitialspace=True)
    table.columns = [c.strip().lower() for c in table.columns]
    if table.empty:
        return 0.0, 0
    score_col = "probability" if "probability" in table.columns else "score"
    return float(table[score_col].max()), int(len(table))


def _run_fpocket(pdb_path: str, _workdir: str) -> tuple[float, int]:
    """Run fpocket and return (best druggability score, pocket count)."""
    subprocess.run(["fpocket", "-f", pdb_path], check=True, capture_output=True, text=True)
    stem = os.path.splitext(pdb_path)[0]
    info_path = f"{stem}_out/{os.path.basename(stem)}_info.txt"
    if not os.path.exists(info_path):
        return 0.0, 0

    scores, n_pockets = [], 0
    with open(info_path) as handle:
        for line in handle:
            if line.strip().startswith("Pocket"):
                n_pockets += 1
            if "Druggability Score" in line:
                try:
                    scores.append(float(line.split(":")[1].strip()))
                except (IndexError, ValueError):
                    continue
    return (max(scores) if scores else 0.0), n_pockets


# Module-level flag to log only the first pocket error (avoids 92k identical warnings)
_first_pocket_error_logged = False


def _detect_pocket(
    accession: str, version: int, tool: str,
    *, prank_exe: Optional[str] = None, java_home: Optional[str] = None,
) -> tuple:
    """Download a model and detect pockets for one accession."""
    global _first_pocket_error_logged
    with tempfile.TemporaryDirectory() as tmp:
        pdb_path = os.path.join(tmp, f"AF-{accession}-F1-model_v{version}.pdb")
        try:
            if not _download_alphafold_model(accession, version, pdb_path):
                return accession, None, 0, tool, "no_model"
            if tool == "prank":
                result = _run_p2rank(
                    pdb_path, tmp, prank_exe=prank_exe or "prank", java_home=java_home,
                )
            else:
                result = _run_fpocket(pdb_path, tmp)
            if result is None:
                return accession, None, 0, tool, "no_output"
            score, n_pockets = result
            return accession, float(score), int(n_pockets), tool, "ok"
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            if not _first_pocket_error_logged:
                LOGGER.error(
                    "[pocket] FIRST FAILURE for %s: %s: %s  "
                    "(suppressing further identical errors)",
                    accession, type(exc).__name__, exc,
                )
                _first_pocket_error_logged = True
            return accession, None, 0, tool, "error"


def _length_proxy_druggability(length: pd.Series) -> pd.Series:
    """Gaussian length proxy used when no pocket tool is available."""
    deviation = (length.astype(float) - POCKET_PROXY_OPTIMUM_LENGTH) / POCKET_PROXY_WIDTH
    return np.exp(-deviation**2).clip(0, 1).round(4)


def add_pocket_druggability(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add binding-pocket druggability from AlphaFold models."""
    df = df.copy()
    tool = "prank" if find_tool("prank") else ("fpocket" if find_tool("fpocket") else None)

    if not cfg.pocket_enable or tool is None:
        df["pocket_druggability"] = _length_proxy_druggability(df["length"])
        df["n_pockets"] = 0
        df["pocket_tool"] = "length_proxy"
        if tool is None and cfg.pocket_enable:
            LOGGER.warning(
                "[pocket] neither p2rank nor fpocket on PATH; using length proxy "
                "(install p2rank to enable real pockets)."
            )
        return df

    # ── Resolve the FULL executable path (Windows can't subprocess .bat by name) ──
    prank_exe = find_tool("prank") if tool == "prank" else None
    if tool == "prank" and prank_exe:
        LOGGER.info("[pocket] resolved prank executable: %s", prank_exe)

    # ── Auto-detect JAVA_HOME if not already set ──
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        java_home = _auto_detect_java_home()
        if java_home:
            LOGGER.info("[pocket] auto-detected JAVA_HOME: %s", java_home)
        else:
            LOGGER.warning(
                "[pocket] JAVA_HOME is not set and could not be auto-detected. "
                "P2Rank will likely fail.  Set JAVA_HOME before running."
            )

    db_path = cfg.path(cfg.pocket_cache_db)
    _init_pocket_cache(db_path)
    with db_connection(db_path) as conn:
        cached = {row[0] for row in conn.execute("SELECT accession FROM pockets")}

    covered = df.loc[df["af_coverage"] == 1, "accession"].tolist()
    if 0 < cfg.pocket_max_structures < len(covered):
        LOGGER.info(
            "[pocket] capping at %d of %d covered structures (--pocket-max).",
            cfg.pocket_max_structures, len(covered),
        )
        covered = covered[: cfg.pocket_max_structures]
    to_run = [acc for acc in covered if acc not in cached]
    LOGGER.info("[pocket] tool=%s | %d cached, %d to compute.", tool, len(cached), len(to_run))

    # ── Smoke-test P2Rank on the first protein BEFORE committing to 92k ──
    if to_run and tool == "prank":
        test_result = _detect_pocket(
            to_run[0], cfg.af_version, tool,
            prank_exe=prank_exe, java_home=java_home,
        )
        if test_result[-1] == "error":
            LOGGER.error(
                "=" * 70 + "\n"
                "[pocket] P2Rank SMOKE TEST FAILED for %s.\n"
                "  prank executable : %s\n"
                "  JAVA_HOME        : %s\n"
                "  Falling back to length-proxy scores.\n"
                "  Fix: set JAVA_HOME and ensure prank.bat is on PATH.\n"
                + "=" * 70,
                to_run[0], prank_exe, java_home,
            )
            df["pocket_druggability"] = _length_proxy_druggability(df["length"])
            df["n_pockets"] = 0
            df["pocket_tool"] = "length_proxy_fallback"
            return df
        LOGGER.info(
            "[pocket] smoke test OK for %s  (score=%s, status=%s)",
            to_run[0], test_result[1], test_result[-1],
        )

    cached_parallel_fetch(
        to_run,
        cache_db=db_path,
        table="pockets",
        fetch_fn=lambda acc: _detect_pocket(
            acc, cfg.af_version, tool, prank_exe=prank_exe, java_home=java_home,
        ),
        workers=cfg.pocket_workers,
        progress_every=200,
        label="pocket",
    )

    with db_connection(db_path) as conn:
        pockets = pd.read_sql(
            "SELECT accession, pocket_score, n_pockets, tool FROM pockets", conn
        )
    df = df.merge(pockets, on="accession", how="left")
    df["pocket_druggability"] = df["pocket_score"].astype(float).fillna(0.0).clip(0, 1).round(4)
    df["n_pockets"] = df["n_pockets"].fillna(0).astype(int)
    df["pocket_tool"] = df["tool"].fillna(tool)
    df = df.drop(columns=["pocket_score", "tool"])
    LOGGER.info("[pocket] median top-pocket score: %.3f", df["pocket_druggability"].median())
    return df
