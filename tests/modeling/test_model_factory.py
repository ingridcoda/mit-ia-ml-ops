"""
test_model_factory.py — Testes de integração atualizados para a arquitetura resiliente.
"""
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling.model_factory import construir_pipeline


@pytest.fixture
def model_cfg_exemplo():
    return {
        'module': 'sklearn.ensemble',
        'class': 'RandomForestClassifier',
        'default_params': {'n_estimators': 10, 'random_state': 42}
    }


@pytest.fixture
def pipe_cfg_com_scaling():
    return {'scaling': {'columns': ['Age', 'Credit amount']}}


@pytest.fixture
def pipe_cfg_sem_scaling():
    return {'scaling': {'columns': []}}


class TestConstruirPipeline:
    def test_construir_pipeline_estrutura_completa(self, model_cfg_exemplo, pipe_cfg_com_scaling):
        """Valida a ordem: Imputer -> Scaler -> Reducer -> Estimator."""
        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=None,
            params_reducer={'strategy': 'pca'},
            pipe_cfg=pipe_cfg_com_scaling
        )

        assert isinstance(pipeline, Pipeline)
        nomes_steps = [step[0] for step in pipeline.steps]

        assert nomes_steps == ['imputer', 'scaler', 'reducer', 'estimator']
        assert isinstance(pipeline.named_steps['imputer'], SimpleImputer)

    def test_construir_pipeline_sem_scaling(self, model_cfg_exemplo, pipe_cfg_sem_scaling):
        """Garante que o scaler é omitido mas o imputer permanece."""
        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=None,
            params_reducer={'strategy': 'passthrough'},
            pipe_cfg=pipe_cfg_sem_scaling
        )

        nomes_steps = [step[0] for step in pipeline.steps]
        assert 'scaler' not in nomes_steps
        assert nomes_steps == ['imputer', 'reducer', 'estimator']
