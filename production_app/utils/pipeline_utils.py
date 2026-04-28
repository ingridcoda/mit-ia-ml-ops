"""
utils/pipeline_utils.py — Preprocessing pipeline for credit inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_UTILS_DIR = Path(__file__).resolve().parent
_APP_DIR = _UTILS_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
_PARQUET_FEATURES = _PROJECT_ROOT / "data" / "features" / "credit_features.parquet"

_FEATURES_TO_KEEP = [
    "Age", "Job", "Credit amount", "log_Credit amount",
    "Duration", "checking_acc_encoded", "is_high_risk_amount"
]


def preprocess_inputs(raw_data: dict) -> pd.DataFrame:
    """Transforms the form dictionary into a DataFrame ready for the model."""
    df = pd.DataFrame([raw_data])

    df["log_Credit amount"] = np.log1p(df["Credit amount"])

    df["is_high_risk_amount"] = (df["Credit amount"] > 5000).astype(int)

    mapa_checking = {"little": 0, "moderate": 1, "rich": 2, "NA": 3}
    df["checking_acc_encoded"] = df["Checking account"].map(mapa_checking).fillna(3)

    df_final = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    return df_final


def get_parquet_features():
    """Returns the path of the parquet file for monitoring."""
    return _PARQUET_FEATURES
