"""
transformers/feature_selector.py — Seletor de Domínio de Crédito.
"""
from __future__ import annotations

import pandas as pd
from src.preprocessing.base import BaseFeatureTransformer


class CreditFeatureSelector(BaseFeatureTransformer):
    """
    Filtra o conjunto de dados para manter apenas as variáveis explicativas
    e o alvo (target) definidos para o modelo de risco.
    """

    def __init__(self, features_to_keep: list[str], strict: bool = False, logger=None):
        super().__init__(logger=logger)
        self.features_to_keep = features_to_keep
        self.strict = strict

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cols_presentes = X.columns.tolist()
        selecionadas = [c for c in self.features_to_keep if c in cols_presentes]
        ausentes = [c for c in self.features_to_keep if c not in cols_presentes]

        if ausentes:
            msg = f"Features ausentes no seletor: {ausentes}"
            if self.strict:
                self._log("ERRO CRÍTICO: %s", msg)
                raise KeyError(msg)
            else:
                self._log("AVISO: %s (Continuando com as disponíveis)", msg)

        self._log("Reduzindo dimensionalidade: %d -> %d colunas.", len(cols_presentes), len(selecionadas))

        return X[selecionadas]
