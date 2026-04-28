"""
test_cross_validation.py — Testes aprimorados para CV Estratificado.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.cross_validation import CVRunner


@pytest.fixture
def dados_sinteticos_binarios() -> tuple[pd.DataFrame, pd.Series]:
    """Gera X e y sintéticos para testes de classificação."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((100, 4)), columns=['age', 'amount', 'duration', 'job'])
    y = pd.Series(rng.integers(0, 2, size=100), name='Risk')
    return X, y


@pytest.fixture
def pipeline_classificacao() -> Pipeline:
    """Gera um pipeline simples para teste do runner."""
    return Pipeline([('estimator', LogisticRegression())])


class TestCVRunnerEvolucao:
    def test_estratificacao_proporcional(self, dados_sinteticos_binarios):
        """Verifica se a proporção de classes é mantida em cada fold."""
        X, y = dados_sinteticos_binarios
        cv_cfg = {'n_splits': 5, 'shuffle': True}
        runner = CVRunner.de_config(cv_cfg, seed=42)
        proporcao_global = y.mean()

        for _, idx_val in runner.cv.split(X, y):
            proporcao_fold = y.iloc[idx_val].mean()
            assert proporcao_fold == pytest.approx(proporcao_global, abs=0.05)

    def test_execucao_com_metricas_completas(self, pipeline_classificacao, dados_sinteticos_binarios):
        """Garante que o runner extrai todas as métricas de cada fold."""
        X, y = dados_sinteticos_binarios
        runner = CVRunner.de_config({'n_splits': 2}, seed=42)
        resultados = runner.executar(pipeline_classificacao, X, y)

        for metrica in ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']:
            assert metrica in resultados[0], f"Métrica {metrica} ausente no fold."

    def test_clonagem_do_modelo(self, dados_sinteticos_binarios):
        """Garante que o modelo original não é alterado pelo fit de um fold."""
        X, y = dados_sinteticos_binarios
        modelo_base = LogisticRegression()
        runner = CVRunner.de_config({'n_splits': 2}, seed=42)
        runner.executar(modelo_base, X, y)
        assert not hasattr(modelo_base, "coef_"), "O modelo original foi treinado! O clone falhou."
