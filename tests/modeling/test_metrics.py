"""
test_metrics.py — Suíte de testes aprimorada para métricas de Classificação de Risco.
"""
import numpy as np
import pytest

from src.modeling.metrics import calcular_metricas, agregar_metricas_folds


class TestCalcularMetricas:
    def test_predicao_perfeita(self) -> None:
        """Predição 100% correta deve retornar todas as métricas como 1.0."""
        y = np.array([0, 1, 0, 1])
        mets = calcular_metricas(y, y)

        assert mets['f1'] == pytest.approx(1.0)
        assert mets['accuracy'] == pytest.approx(1.0)
        assert mets['precision'] == pytest.approx(1.0)
        assert mets['recall'] == pytest.approx(1.0)

    def test_predicao_totalmente_errada(self) -> None:
        """Predição 100% invertida deve resultar em métricas nulas."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        mets = calcular_metricas(y_true, y_pred)

        assert mets['f1'] == pytest.approx(0.0)
        assert mets['accuracy'] == pytest.approx(0.0)
        assert mets['precision'] == pytest.approx(0.0)
        assert mets['recall'] == pytest.approx(0.0)

    def test_zero_division_safety(self) -> None:
        """Garante que se o modelo não prever nenhuma classe positiva, não haja erro de divisão por zero."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # Nenhuma classe positiva prevista

        mets = calcular_metricas(y_true, y_pred)

        # Graças ao zero_division=0 no metrics.py, deve retornar 0.0 sem estourar erro
        assert mets['precision'] == 0.0
        assert mets['recall'] == 0.0

    def test_roc_auc_condicional(self) -> None:
        """Verifica se o ROC-AUC só é incluído quando probabilidades são fornecidas."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])

        # Caso com probabilidades
        mets_com_prob = calcular_metricas(y_true, y_true, y_prob)
        assert 'roc_auc' in mets_com_prob
        assert mets_com_prob['roc_auc'] == pytest.approx(1.0)

        # Caso sem probabilidades
        mets_sem_prob = calcular_metricas(y_true, y_true)
        assert 'roc_auc' not in mets_sem_prob


class TestAgregarMetricasFolds:
    def test_agregacao_estatistica_completa(self) -> None:
        """Verifica se a média e o desvio padrão de múltiplos folds são calculados corretamente."""
        fold_mets = [
            {'f1': 0.70, 'accuracy': 0.80},
            {'f1': 0.90, 'accuracy': 0.90}
        ]
        agg = agregar_metricas_folds(fold_mets)

        # Médias: (0.7 + 0.9)/2 = 0.8 | (0.8 + 0.9)/2 = 0.85
        assert agg['cv_f1_mean'] == pytest.approx(0.8)
        assert agg['cv_accuracy_mean'] == pytest.approx(0.85)

        # Desvios padrão (std deve ser maior que zero)
        assert agg['cv_f1_std'] > 0
        assert agg['cv_accuracy_std'] > 0

    def test_agregacao_ignora_campo_fold(self) -> None:
        """Garante que a chave 'fold' é descartada e não gera métricas de média para ela."""
        fold_mets = [
            {'fold': 1, 'f1': 0.8},
            {'fold': 2, 'f1': 0.9}
        ]
        agg = agregar_metricas_folds(fold_mets)

        assert 'cv_f1_mean' in agg
        assert 'cv_fold_mean' not in agg  # Deve ignorar a coluna 'fold'
