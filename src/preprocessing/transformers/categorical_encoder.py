"""transformers/categorical_encoder.py — Generic Encoder."""
from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class CategoricalEncoder(BaseFeatureTransformer):
    """Applies One-Hot and Ordinal encoding to credit variables."""

    def __init__(self, enc_config: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.enc_config = enc_config

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for config in self.enc_config:
            col = config.get("column")
            if col in X.columns:

                if "ordinal_map" in config:
                    X[config["ordinal_column"]] = X[col].map(config["ordinal_map"])

                dummies = pd.get_dummies(X[col], prefix=config.get("one_hot_prefix", col))
                X = pd.concat([X, dummies.astype(int)], axis=1)
        return X
