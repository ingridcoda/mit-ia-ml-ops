"""transformers/ratio_features.py — Razões Financeiras."""
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class RatioFeatureTransformer(BaseFeatureTransformer):
    """Cria razões (A / B). Ex: Credit Amount / Duration."""

    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.ratios = ratios

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.ratios:
            name, num, den = spec["name"], spec["numerator"], spec["denominator"]
            if num in X.columns and den in X.columns:
                X[name] = (X[num] / X[den].replace(0, np.nan))
                self._log("Ratio '%s' criado com sucesso.", name)
        return X
