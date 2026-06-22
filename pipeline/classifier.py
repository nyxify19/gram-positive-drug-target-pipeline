"""ML druggability classifier (XGBoost / Random Forest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import Config, FEATURE_COLS, LOGGER


def assign_druggable_label(df: pd.DataFrame) -> pd.DataFrame:
    """Label proteins as druggable from multiple orthogonal evidence sources."""
    df = df.copy()

    has_drugbank = df["drugbank_targets"].str.len() > 0
    has_pdb = df["pdb_structures"].str.len() > 0
    is_enzyme = df["protein_name"].str.contains(
        r"(?i)\b(?:ase|kinase|synthase|transferase|polymerase|reductase|oxidase|protease|ligase|hydrolase)\b",
        regex=True, na=False,
    )
    has_keyword = df["keywords"].str.contains(
        r"(?i)antibiotic|antimicrobial|drug.?target|virulence|toxin|pathogen",
        regex=True, na=False,
    )
    has_virulence = df["function"].str.contains(
        r"(?i)virulence|pathogen|invasion|adhesion|toxin|hemolysin",
        regex=True, na=False,
    )
    has_function_text = df["function"].str.contains(
        r"(?i)essential|required for growth|lethal|cell.?wall|peptidoglycan|"
        r"dna replication|transcription|translation|ribosom",
        regex=True, na=False,
    )

    df["druggable_label"] = (
        has_drugbank | (has_pdb & is_enzyme) | has_keyword | has_virulence | has_function_text
    ).astype(int)

    n_pos = int(df["druggable_label"].sum())
    LOGGER.info(
        "[label] druggable positives: %d / %d (%.1f%%)",
        n_pos, len(df), 100 * n_pos / max(len(df), 1),
    )
    return df


def _find_f1_threshold(labels: pd.Series, proba: np.ndarray) -> float:
    """Return the probability cut-off that maximises F1 on the OOF predictions."""
    precision, recall, thresholds = precision_recall_curve(labels, proba)
    if not len(thresholds):
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    if not np.isfinite(f1).any():
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def _permutation_importances(
    features: pd.DataFrame, labels: pd.Series, base_estimator, seed: int
) -> pd.Series:
    """Permutation importances on a held-out split (honest generalisation signal)."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=seed
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_estimator)]).fit(x_train, y_train)
    result = permutation_importance(
        pipe, x_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=seed, n_jobs=-1,
    )
    return pd.Series(
        result.importances_mean, index=FEATURE_COLS
    ).sort_values(ascending=False)


def _build_classifier(minority: int, majority: int, calibration_folds: int, seed: int):
    """Return the best available classifier for severe class imbalance."""
    imbalance_ratio = majority / max(minority, 1)
    calibration_method = "isotonic" if minority >= 50 else "sigmoid"

    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
        LOGGER.info(
            "[model] using XGBoost (scale_pos_weight=%.1f, calibration=%s)",
            imbalance_ratio, calibration_method,
        )
        base = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    except ImportError:
        LOGGER.info(
            "[model] xgboost not installed; using RandomForest "
            "(pip install xgboost for better imbalance handling)"
        )
        base = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])

    classifier = CalibratedClassifierCV(pipe, method=calibration_method, cv=calibration_folds)
    return classifier, base


def train_druggability_model(df: pd.DataFrame, cfg: Config) -> tuple:
    """Train and cross-validate a druggability model tuned for extreme class imbalance."""
    df = assign_druggable_label(df)
    features = df[FEATURE_COLS].astype(float).fillna(0.0)
    labels = df["druggable_label"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Only one class present; cannot train classifier.")

    counts = labels.value_counts()
    minority = int(counts.min())
    majority = int(counts.max())
    n_splits = max(2, min(5, minority))
    calibration_folds = max(2, min(3, minority))
    if n_splits < 5:
        LOGGER.warning("Small minority class (%d); using %d CV folds.", minority, n_splits)
    LOGGER.info(
        "[model] class counts: positive=%d, negative=%d, imbalance ratio=%.1f:1",
        minority, majority, majority / max(minority, 1),
    )

    classifier, base_estimator = _build_classifier(minority, majority, calibration_folds, cfg.seed)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    proba = cross_val_predict(
        classifier, features, labels, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = _find_f1_threshold(labels, proba)
    y_pred = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(labels, proba)
    pr_auc = average_precision_score(labels, proba)
    baseline = labels.mean()
    LOGGER.info(
        "CV calibrated ROC-AUC: %.3f | PR-AUC: %.3f (baseline %.3f) | F1 threshold %.3f",
        roc_auc, pr_auc, baseline, threshold,
    )
    LOGGER.info("\n%s", classification_report(labels, y_pred, zero_division=0))

    df = df.copy()
    df["druggability_proba"] = proba
    importances = _permutation_importances(features, labels, base_estimator, cfg.seed)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "baseline_rate": float(baseline),
        "operating_threshold": float(threshold),
    }
    return df, classifier, metrics, importances, y_pred, labels
