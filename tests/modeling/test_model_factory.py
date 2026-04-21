"""
test_model_factory.py — Testes de integração para a Fábrica de Pipelines.
"""
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.modeling.model_factory import construir_pipeline
from src.modeling.reducer import FeatureReducer
from src.preprocessing.transformers.stateful import StandardScalerTransformer


# ── FIXTURES PARA CONFIGURAÇÃO ──────────────────────────────────────────────

@pytest.fixture
def model_cfg_exemplo():
    """Configuração base para um classificador RandomForest."""
    return {
        'module': 'sklearn.ensemble',
        'class': 'RandomForestClassifier',
        'default_params': {'n_estimators': 10, 'random_state': 42}
    }


@pytest.fixture
def pipe_cfg_com_scaling():
    """Configuração de pipeline com colunas para escalonamento."""
    return {'scaling': {'columns': ['Age', 'Credit amount']}}


@pytest.fixture
def pipe_cfg_sem_scaling():
    """Configuração de pipeline sem colunas de escalonamento."""
    return {'scaling': {'columns': []}}


# ── TESTES DE COBERTURA ─────────────────────────────────────────────────────

class TestConstruirPipeline:
    def test_construir_pipeline_estrutura_completa(self, model_cfg_exemplo, pipe_cfg_com_scaling):
        """Garante que o pipeline tem os 3 passos corretos quando o scaling está ativo."""
        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=None,
            params_reducer={'strategy': 'pca'},
            pipe_cfg=pipe_cfg_com_scaling
        )

        assert isinstance(pipeline, Pipeline)
        # Verifica nomes e ordens dos steps: Scaler -> Reducer -> Estimator
        nomes_steps = [step[0] for step in pipeline.steps]
        assert nomes_steps == ['scaler', 'reducer', 'estimator']

        # Verifica se as classes corretas foram instanciadas
        assert isinstance(pipeline.named_steps['scaler'], StandardScalerTransformer)
        assert isinstance(pipeline.named_steps['reducer'], FeatureReducer)
        assert isinstance(pipeline.named_steps['estimator'], RandomForestClassifier)

    def test_construir_pipeline_sem_scaling(self, model_cfg_exemplo, pipe_cfg_sem_scaling):
        """Garante que o passo de scaler é omitido quando não há colunas configuradas."""
        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=None,
            params_reducer={'strategy': 'passthrough'},
            pipe_cfg=pipe_cfg_sem_scaling
        )

        nomes_steps = [step[0] for step in pipeline.steps]
        assert 'scaler' not in nomes_steps
        assert nomes_steps == ['reducer', 'estimator']

    def test_sobrescrita_de_parametros_do_modelo(self, model_cfg_exemplo, pipe_cfg_sem_scaling):
        """Valida se os parâmetros dinâmicos (ex: do Optuna) sobrescrevem os padrões."""
        params_dinamicos = {'n_estimators': 100, 'max_depth': 5}

        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=params_dinamicos,
            params_reducer=None,
            pipe_cfg=pipe_cfg_sem_scaling
        )

        estimator = pipeline.named_steps['estimator']
        # Deve usar o valor dinâmico (100) e não o padrão da fixture (10)
        assert estimator.n_estimators == 100
        assert estimator.max_depth == 5

    def test_passagem_de_parametros_para_o_reducer(self, model_cfg_exemplo, pipe_cfg_sem_scaling):
        """Garante que as configurações de redução chegam ao FeatureReducer."""
        params_reducer = {'strategy': 'lda', 'n_components': 1}

        pipeline = construir_pipeline(
            model_cfg=model_cfg_exemplo,
            params_modelo=None,
            params_reducer=params_reducer,
            pipe_cfg=pipe_cfg_sem_scaling
        )

        reducer = pipeline.named_steps['reducer']
        assert reducer.strategy == 'lda'
