"""
transformers/stateful.py — Transformadores com Memória (Stateful).

Estes transformadores aprendem estatísticas (média, mediana, desvio padrão)
durante o fit() no conjunto de treinamento.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.preprocessing.base import BaseFeatureTransformer


class GroupWiseImputer(BaseFeatureTransformer):
    """
    Imputador que utiliza estatísticas de grupo para preencher valores nulos.
    No crédito, isso é vital para preencher 'Saving accounts' ou 'Checking account'
    baseado no perfil do 'Job' ou 'Age_Category'.
    """

    def __init__(self, group_col: str, target_col: str, strategy: str = "median", logger=None):
        super().__init__(logger=logger)
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.learned_values_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> GroupWiseImputer:
        self._log("Calculando %s de '%s' por grupo '%s'...", self.strategy, self.target_col, self.group_col)

        if self.strategy == "median":
            stats = X.groupby(self.group_col)[self.target_col].median()
        else:
            stats = X.groupby(self.group_col)[self.target_col].mean()

        self.learned_values_ = stats.to_dict()
        self.global_fallback_ = X[self.target_col].median() if self.strategy == "median" else X[self.target_col].mean()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Preenche baseado no mapa aprendido; se o grupo for novo, usa o fallback global
        X[self.target_col] = X[self.target_col].fillna(X[self.group_col].map(self.learned_values_))
        X[self.target_col] = X[self.target_col].fillna(self.global_fallback_)

        self._log("Imputação de '%s' concluída.", self.target_col)
        return X


class ZScoreScaler(BaseFeatureTransformer):
    """
    Escalonador Z-Score (Standardization).
    Transforma features para média 0 e desvio padrão 1.
    """

    def __init__(self, columns: list[str], logger=None):
        super().__init__(logger=logger)
        self.columns = columns
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> ZScoreScaler:
        for col in self.columns:
            if col in X.columns:
                mu = X[col].mean()
                sigma = X[col].std()
                # Proteção contra variância zero
                self.stats_[col] = {"mean": mu, "std": sigma if sigma > 0 else 1.0}

        self._log("Parâmetros de escala aprendidos para %d colunas.", len(self.stats_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, s in self.stats_.items():
            if col in X.columns:
                X[col] = (X[col] - s["mean"]) / s["std"]
        return X


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper para o StandardScaler do sklearn que lida com DataFrames e
    mantém a compatibilidade com o Pipeline de ML.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Define quais colunas serão escalonadas
        self.cols_to_fit_ = self.columns if self.columns else X.columns
        self.scaler.fit(X[self.cols_to_fit_])
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Aplica a transformação apenas nas colunas selecionadas
        X_copy[self.cols_to_fit_] = self.scaler.transform(X[self.cols_to_fit_])
        return X_copy
