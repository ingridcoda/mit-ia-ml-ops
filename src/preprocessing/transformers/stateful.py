"""
transformers/stateful.py — Transformadores com Memória (Stateful).
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.preprocessing.base import BaseFeatureTransformer


class GroupWiseImputer(BaseFeatureTransformer):
    def __init__(self, group_col: str, target_col: str, strategy: str = "median", logger=None):
        super().__init__(logger=logger)
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.learned_values_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> GroupWiseImputer:
        if self.strategy == "median":
            stats = X.groupby(self.group_col)[self.target_col].median()
        else:
            stats = X.groupby(self.group_col)[self.target_col].mean()
        self.learned_values_ = stats.to_dict()
        self.global_fallback_ = X[self.target_col].median() if self.strategy == "median" else X[self.target_col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.target_col] = X[self.target_col].fillna(X[self.group_col].map(self.learned_values_))
        X[self.target_col] = X[self.target_col].fillna(self.global_fallback_)
        return X


class ZScoreScaler(BaseFeatureTransformer):
    def __init__(self, columns: list[str], logger=None):
        super().__init__(logger=logger)
        self.columns = columns
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> ZScoreScaler:
        for col in self.columns:
            if col in X.columns:
                mu = X[col].mean()
                sigma = X[col].std()
                self.stats_[col] = {"mean": mu, "std": sigma if sigma > 0 else 1.0}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, s in self.stats_.items():
            if col in X.columns:
                X[col] = (X[col] - s["mean"]) / s["std"]
        return X


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):

        if not hasattr(X, "columns") and self.columns:
            X = pd.DataFrame(X, columns=self.columns)
        self.cols_to_fit_ = self.columns if self.columns else X.columns
        self.scaler.fit(X[self.cols_to_fit_])
        return self

    def transform(self, X):
        if not hasattr(X, "columns") and self.columns:
            X = pd.DataFrame(X, columns=self.columns)
        X_copy = X.copy()
        X_copy[self.cols_to_fit_] = self.scaler.transform(X[self.cols_to_fit_])
        return X_copy
