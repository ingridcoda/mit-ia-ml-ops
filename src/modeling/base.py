"""
modeling/base.py — Base classes for optimization and evaluation in the modeling module.
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
        pass


class BaseEvaluator(ABC):
    @abstractmethod
    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        pass
