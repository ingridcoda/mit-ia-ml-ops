"""transformers/log_transform.py — Applies logarithmic transformation to specified numeric columns to reduce skewness."""
from typing import Any

import numpy as np
import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer


class LogTransformer(BaseFeatureTransformer):
    def __init__(self, columns: list[str], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.columns = columns

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
                self._log("Log transform aplicado em: %s", col)
        return X
