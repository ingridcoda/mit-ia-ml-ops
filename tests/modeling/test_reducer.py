"""
test_reducer.py — Testes unitários para o redutor de dimensionalidade.
"""
import numpy as np
import pandas as pd
import pytest

from src.modeling.reducer import FeatureReducer


@pytest.fixture
def dummy_data():
    """Gera dados sintéticos para teste de redução."""
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f'col_{i}' for i in range(10)])
    y = pd.Series([0, 1] * 10)
    return X, y


class TestFeatureReducer:
    def test_estrategia_passthrough(self, dummy_data):
        """Garante que 'passthrough' retorna o DataFrame original intacto."""
        X, _ = dummy_data
        reducer = FeatureReducer(strategy='passthrough')
        X_transformed = reducer.fit_transform(X)

        pd.testing.assert_frame_equal(X_transformed, X)
        assert X_transformed.shape[1] == 10

    def test_estrategia_pca(self, dummy_data):
        """Verifica se o PCA reduz para o número fixo de componentes solicitado."""
        X, _ = dummy_data
        reducer = FeatureReducer(strategy='pca', n_components=2)
        X_transformed = reducer.fit_transform(X)

        assert X_transformed.shape[1] == 2
        assert isinstance(X_transformed, np.ndarray)

    def test_estrategia_lda_binario(self, dummy_data):
        """Garante que o LDA reduz para 1 dimensão em problemas binários (Good/Bad)."""
        X, y = dummy_data
        reducer = FeatureReducer(strategy='lda')
        X_transformed = reducer.fit_transform(X, y)

        assert X_transformed.shape[1] == 1

    def test_reducer_nao_treinado(self, dummy_data):
        """Valida que o transform não quebra caso o fit não tenha sido chamado corretamente."""
        X, _ = dummy_data
        reducer = FeatureReducer(strategy='pca')

        assert reducer.transform(X).shape == X.shape
