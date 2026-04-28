"""
transformers/polynomial_features.py — Interaction and Polynomial Transformer.

Creates interaction terms (x1 * x2) or quadratic terms (x1^2) to capture
complexities in credit behavior that simple linear models would miss.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer


class FeatureInteractionTransformer(BaseFeatureTransformer):
    """
    Generates new features based on the interaction between existing columns.

    Examples in Credit:
    - Age * Credit Amount : Captures the risk of high loans for young people.
    - Credit Amount * Duration : Represents the total volume of financial exposure.
    """

    def __init__(self, interaction_config: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.interaction_config = interaction_config

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executes the multiplications configured in the preprocessing.yaml."""
        X = X.copy()

        for spec in self.interaction_config:
            new_feature_name = spec["name"]
            input_cols = spec["columns"]

            missing = [c for c in input_cols if c not in X.columns]
            if missing:
                self._warn("Failed to create '%s': columns %s not found.", new_feature_name, missing)
                continue

            if not all(pd.api.types.is_numeric_dtype(X[c]) for c in input_cols):
                self._warn("Failed to create '%s': one or more columns are not numeric.", new_feature_name)
                continue

            try:
                if len(input_cols) == 1:

                    X[new_feature_name] = X[input_cols[0]] ** 2
                elif len(input_cols) == 2:

                    X[new_feature_name] = X[input_cols[0]] * X[input_cols[1]]
                else:
                    self._warn("Ignoring '%s': we only support 1 or 2 columns per interaction.", new_feature_name)
                    continue

                self._log("Interaction feature '%s' successfully generated.", new_feature_name)

            except Exception as e:
                self._warn("Unexpected error processing '%s': %s", new_feature_name, str(e))

        return X
