"""
test_evaluator.py — Suíte completa de testes para validação e robustez.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from src.modeling.evaluator import HoldoutEvaluator


def test_diagnostico_robustez_boa(null_logger):
    """Garante que modelos com métricas próximas são considerados robustos."""
    evaluator = HoldoutEvaluator(null_logger)

    status = evaluator.diagnosticar_robustez(cv_f1=0.80, holdout_f1=0.78)
    assert status == "BOA"


def test_diagnostico_robustez_overfitting(null_logger):
    """Garante a detecção de queda acentuada de performance no teste."""
    evaluator = HoldoutEvaluator(null_logger)

    status = evaluator.diagnosticar_robustez(cv_f1=0.90, holdout_f1=0.70)
    assert status == "RUIM (Risco de Overfitting)"


def test_buscar_melhor_threshold_logica(null_logger):
    """Verifica se o buscador encontra um limiar que maximiza o F1."""
    evaluator = HoldoutEvaluator(null_logger)

    y_true = np.array([0, 0, 1, 1])

    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, score = evaluator.buscar_melhor_threshold(y_true, y_prob)

    assert 0.2 < threshold < 0.8
    assert score == 1.0


def test_avaliar_fluxo_com_probabilidade(null_logger):
    """Testa o caminho principal: modelo que suporta predict_proba."""
    evaluator = HoldoutEvaluator(null_logger)

    model = MagicMock(spec=['predict_proba'])

    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])

    X = pd.DataFrame({'feat': [1, 2]})
    y = pd.Series([0, 1])

    metrics = evaluator.avaliar(model, X, y)

    assert 'best_threshold' in metrics
    assert metrics['f1'] == 1.0
    model.predict_proba.assert_called_once()


def test_avaliar_fluxo_sem_probabilidade(null_logger):
    """Garante que o avaliador funciona com modelos determinísticos (sem proba)."""
    evaluator = HoldoutEvaluator(null_logger)

    model = MagicMock(spec=['predict'])
    model.predict.return_value = np.array([0, 1])

    X = pd.DataFrame({'feat': [1, 2]})
    y = pd.Series([0, 1])

    metrics = evaluator.avaliar(model, X, y)

    assert 'best_threshold' not in metrics
    assert metrics['f1'] == 1.0
    model.predict.assert_called_once()
