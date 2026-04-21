"""
modeling/evaluator.py — Avaliador de Generalização e Otimização de Limiar.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.modeling.base import BaseEvaluator
from src.modeling.metrics import calcular_metricas


class HoldoutEvaluator(BaseEvaluator):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def buscar_melhor_threshold(self, y_true, y_prob):
        """Maximiza o F1-Score variando o limiar de decisão."""
        thresholds = np.linspace(0.1, 0.9, 80)
        scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        melhor_t = thresholds[np.argmax(scores)]
        return melhor_t, np.max(scores)

    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """Avalia o pipeline e retorna métricas com o melhor threshold encontrado."""
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        if y_prob is not None:
            melhor_t, melhor_f1 = self.buscar_melhor_threshold(y.values, y_prob)
            y_pred = (y_prob >= melhor_t).astype(int)

            metrics = calcular_metricas(y.values, y_pred, y_prob)
            metrics['best_threshold'] = float(melhor_t)
            return metrics
        else:
            y_pred = model.predict(X)
            return calcular_metricas(y.values, y_pred, y_prob)

    def diagnosticar_robustez(self, cv_f1: float, holdout_f1: float) -> str:
        """Compara métricas de treino/validação para diagnosticar overfitting."""
        delta_pct = abs(holdout_f1 - cv_f1) / cv_f1 * 100
        return 'BOA' if delta_pct < 15 else 'RUIM (Risco de Overfitting)'
