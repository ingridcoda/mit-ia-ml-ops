"""transformers/categorical_encoder.py — Encoder Genérico."""
from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class CategoricalEncoder(BaseFeatureTransformer):
    """Aplica One-Hot e Ordinal encoding em variáveis de crédito."""

    def __init__(self, enc_config: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.enc_config = enc_config

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for config in self.enc_config:
            col = config.get("column")
            if col in X.columns:
                # 1. Ordinal (se houver mapa)
                if "ordinal_map" in config:
                    X[config["ordinal_column"]] = X[col].map(config["ordinal_map"])
                # 2. One-Hot
                dummies = pd.get_dummies(X[col], prefix=config.get("one_hot_prefix", col))
                X = pd.concat([X, dummies.astype(int)], axis=1)
        return X
