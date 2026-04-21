"""
preprocessing/base.py — Classe abstrata para transformadores de Crédito.

Centraliza o comportamento comum (logging e interface sklearn).
Segue o princípio de segregação de interfaces.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Classe base para todos os transformadores do pipeline de risco.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> BaseFeatureTransformer:
        """Padrão para transformadores stateless."""
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Contrato de transformação de dados."""
