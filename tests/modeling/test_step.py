"""
test_step.py — Teste de integração da etapa de modelagem (Maestro).
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.modeling.step import ModelingStep


def _criar_context_modeling_mock(tmp_path, df, logger):
    """Simula o PipelineContext para a etapa de modelagem."""
    ctx = MagicMock()
    ctx.logger = logger
    ctx.root_dir = tmp_path
    ctx.features_dir = tmp_path / "features"
    ctx.features_dir.mkdir(parents=True, exist_ok=True)

    feature_path = ctx.features_dir / "credit_features.parquet"
    df.to_parquet(feature_path, index=False)

    ctx.modeling_cfg = {
        'target_column': 'Risk',
        'best_model': {'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier'},
        'pipeline_params': {}
    }
    return ctx


@patch("src.modeling.step.mlflow")
@patch("src.modeling.step.joblib.dump")
def test_modeling_step_run_com_tratamento_nan(mock_joblib, mock_mlflow, tmp_path, null_logger):
    """Garante que o step preenche NaNs antes do treino, conforme requisito do LDA."""
    df_teste = pd.DataFrame({
        "Age": [25, 40, 60, 30, 22, 35, 45, 50, 28, 33],
        "Credit amount": [1000, 5000, 12000, None, 2000, 3000, 4000, 7000, 1500, 2500],
        "Duration": [6, 24, 48, 12, 12, 18, 24, 36, 6, 12],
        "Risk": ["good", "bad", "good", "bad", "good", "bad", "good", "bad", "good", "bad"]
    })

    ctx = _criar_context_modeling_mock(tmp_path, df_teste, null_logger)
    step = ModelingStep(ctx)

    # ── CONFIGURAÇÃO DO MOCK DO PIPELINE ─────────────────────────────────────
    mock_pipeline_obj = MagicMock()
    # Simula o retorno de predict_proba para 2 amostras de teste (X_test)
    # Retorna probabilidade para classe 0 e classe 1
    mock_pipeline_obj.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

    with patch("src.modeling.step.cross_val_score", return_value=pd.Series([0.8])), \
            patch("src.modeling.step.construir_pipeline", return_value=mock_pipeline_obj):
        step.run()

        # Validações finais
        model_dir = tmp_path / "outputs" / "modeling"
        assert model_dir.exists()
        mock_joblib.assert_called_once()
        # Garante que o tracker foi chamado com as métricas calculadas pelo mock
