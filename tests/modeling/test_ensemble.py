"""
test_ensemble.py — Testes para o construtor de modelos Ensemble.
"""
from unittest.mock import MagicMock, patch

from sklearn.ensemble import VotingClassifier

from src.modeling.ensemble import EnsembleBuilder


def test_build_voting_estrutura_e_metricas():
    """Garante que o voting classifier é instanciado e as métricas são agregadas."""
    builder = EnsembleBuilder()

    builder.cv_runner = MagicMock()
    builder.cv_runner.executar.return_value = [{'f1': 0.8}, {'f1': 0.85}]
    builder._construir_estimadores_base = MagicMock(return_value=[('m1', MagicMock()), ('m2', MagicMock())])

    with patch('src.modeling.ensemble.agregar_metricas_folds') as mock_agg:
        mock_agg.return_value = {'cv_f1_mean': 0.825}

        result = builder.build_voting(top_n_entries=[], X_train=None, y_train=None)

        assert isinstance(result['_instance'], VotingClassifier)
        assert result['cv_f1_mean'] == 0.825
