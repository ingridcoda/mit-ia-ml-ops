"""
modeling/cross_validation.py — Executor de Cross-Validation Estratificado.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.modeling.metrics import calcular_metricas


class CVRunner:
    def __init__(self, cv: StratifiedKFold, logger: logging.Logger | None = None) -> None:
        self.cv = cv
        self.logger = logger

    def executar(self, modelo: Any, X: pd.DataFrame, y: pd.Series) -> list[dict]:
        metricas_folds = []

        for i_fold, (idx_treino, idx_val) in enumerate(self.cv.split(X, y)):
            m = clone(modelo)
            m.fit(X.iloc[idx_treino], y.iloc[idx_treino])

            y_pred = m.predict(X.iloc[idx_val])

            y_prob = m.predict_proba(X.iloc[idx_val])[:, 1] if hasattr(m, "predict_proba") else None

            metricas = calcular_metricas(y.iloc[idx_val].values, y_pred, y_prob)
            metricas['fold'] = i_fold + 1
            metricas_folds.append(metricas)
        return metricas_folds

    @staticmethod
    def de_config(cv_cfg: dict, seed: int) -> "CVRunner":
        cv = StratifiedKFold(
            n_splits=cv_cfg.get('n_splits', 5),
            shuffle=cv_cfg.get('shuffle', True),
            random_state=seed
        )
        return CVRunner(cv)
