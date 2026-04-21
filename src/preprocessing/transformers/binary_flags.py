"""transformers/binary_flags.py — Flags para valores de risco."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class BinaryFlagTransformer(BaseFeatureTransformer):
    """Cria colunas binárias (0/1) para valores de corte (thresholds)."""

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
