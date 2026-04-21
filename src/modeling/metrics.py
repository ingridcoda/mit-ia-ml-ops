"""
modeling/metrics.py — Funções de cálculo e agregação de métricas de classificação.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """Calcula métricas essenciais para Risco de Crédito."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    return metrics


def agregar_metricas_folds(fold_metrics: list[dict]) -> dict:
    """Agrega métricas de CV em média ± desvio padrão."""
    df = pd.DataFrame(fold_metrics)
    res = {}
    for col in df.columns:
        if col == 'fold': continue
        res[f'cv_{col}_mean'] = float(df[col].mean())
        res[f'cv_{col}_std'] = float(df[col].std())
    return res
