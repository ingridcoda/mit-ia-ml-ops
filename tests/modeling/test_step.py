"""
test_step.py — Teste de integração da etapa de modelagem (Maestro).
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.modeling.step import ModelingStep


def _criar_context_modeling_mock(tmp_path, df, logger):
    ctx = MagicMock()
    ctx.logger = logger
    ctx.root_dir = tmp_path
    ctx.features_dir = tmp_path / "features"
    ctx.features_dir.mkdir(parents=True, exist_ok=True)

    feature_path = ctx.features_dir / "credit_features.parquet"
    df.to_parquet(feature_path, index=False)

    ctx.modeling_cfg = {
        'target_column': 'Risk',
        'best_model': {'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier', 'params': {}},
        'pipeline_params': {}
    }
    return ctx


@patch("src.modeling.step.MLflowTracker")
@patch("src.modeling.step.mlflow")
@patch("src.modeling.step.joblib.dump")
def test_modeling_step_run_com_tratamento_nan(mock_joblib, mock_mlflow, mock_tracker_class, tmp_path, null_logger):
    df_teste = pd.DataFrame({
        "Age": [25, 40, 60, 30, 22, 35, 45, 50, 28, 33],
        "Credit amount": [1000, 5000, 12000, 1000, 2000, 3000, 4000, 7000, 1500, 2500],
        "Duration": [6, 24, 48, 12, 12, 18, 24, 36, 6, 12],
        "Risk": ["good", "bad", "good", "bad", "good", "bad", "good", "bad", "good", "bad"]
    })

    ctx = _criar_context_modeling_mock(tmp_path, df_teste, null_logger)
    ctx.params = {'current_version': 'v1', 'selected_reducer': 'lda'}
    step = ModelingStep(ctx)

    with patch("src.modeling.step.construir_pipeline") as mock_build, patch(
            "src.modeling.step.cross_val_score") as mock_cv, patch(
        "src.modeling.step.HoldoutEvaluator") as mock_evaluator:
        mock_pipeline = MagicMock()
        mock_build.return_value = mock_pipeline
        mock_cv.return_value = np.array([0.8, 0.85])

        mock_eval_inst = mock_evaluator.return_value
        mock_eval_inst.avaliar.return_value = {"f1": 0.85, "accuracy": 0.9}
        mock_eval_inst.diagnosticar_robustez.return_value = "Normal"

        mock_tracker_inst = mock_tracker_class.return_value

        step.run()

        assert mock_build.called
        args, kwargs = mock_build.call_args
        assert "model_cfg" in kwargs
        assert "params_reducer" in kwargs
        assert "params_modelo" in kwargs
        assert "pipe_cfg" in kwargs

        assert mock_pipeline.fit.called
        mock_tracker_inst.salvar_modelo.assert_called_once()
