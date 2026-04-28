"""transformers/binary_flags.py — Creates binary (0/1) columns based on specified thresholds for given features."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class BinaryFlagTransformer(BaseFeatureTransformer):
    def __init__(self, flags: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.flags = flags

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.flags:
            col, val, new_col = spec["column"], spec["value"], spec["new_column"]
            if col in X.columns:
                X[new_col] = (X[col] >= val).astype(int)
                self._log("Flag '%s': %d linhas ativadas.", new_col, int(X[new_col].sum()))
        return X
