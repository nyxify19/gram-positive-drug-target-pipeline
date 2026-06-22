"""Reproducibility manifest writer."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.config import COMPOSITE_WEIGHTS, Config, FEATURE_COLS, LOGGER


def write_manifest(
    cfg: Config, df: pd.DataFrame, metrics: dict, feature_report: dict
) -> None:
    """Write a JSON manifest capturing inputs, environment and results."""
    import Bio
    import sklearn

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "uniprot_release": df.attrs.get("uniprot_release", "unknown"),
        "n_proteins": int(len(df)),
        "config": asdict(cfg),
        "metrics": metrics,
        "tools": {
            tool: (shutil.which(tool) or None) for tool in ("mmseqs", "prank", "fpocket")
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "biopython": Bio.__version__,
        },
        "composite_weights": COMPOSITE_WEIGHTS,
        "feature_cols": FEATURE_COLS,
        "feature_availability": feature_report,
        "tier_distribution": (
            df["priority_tier"].value_counts().to_dict()
            if "priority_tier" in df.columns else {}
        ),
    }
    with open(cfg.path("run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    LOGGER.info("Manifest written: %s", cfg.path("run_manifest.json"))
