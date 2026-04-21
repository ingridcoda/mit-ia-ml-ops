"""
modeling/base.py — Classes abstratas base do módulo de modelagem.

Define os contratos principais para Classificação de Risco:
  - BaseOptimizer    : Estratégias de busca (Optuna, GridSearch).
  - BaseEvaluator    : Avaliadores (Holdout, Cross-Validation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    @abstractmethod
    def otimizar(
            self,
            model_name: str,
            model_cfg: dict,
            X_tune: Any,
            y_tune: Any,
            pipe_cfg: dict,
            feat_red_cfg: dict,
    ) -> dict:
        """Executa a otimização e retorna os melhores parâmetros."""


class BaseEvaluator(ABC):
    @abstractmethod
    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """
        Avalia o modelo de classificação e retorna métricas:
        accuracy, precision, recall, f1, roc_auc.
        """
