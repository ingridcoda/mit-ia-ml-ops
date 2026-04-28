"""
test_tracker.py — Suíte completa de testes para o rastreador de experiências.
"""
import json
from unittest.mock import patch, MagicMock

from src.modeling.tracker import MLflowTracker


class TestMLflowTracker:
    @patch("mlflow.log_artifact")
    @patch("mlflow.set_tag")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metric")
    def test_log_experimento_fluxo_completo(self, mock_metric, mock_params, mock_tag, mock_artifact):
        tracker = MLflowTracker()
        metrics = {"f1": 0.82, "status": "sucesso"}

        params = {"n_estimators": 100, "reducer": "pca"}
        artifacts = {"modelo": "caminho/modelo.joblib"}

        tracker.log_experimento("XGBoost", metrics, params, artifacts)

        mock_metric.assert_called_once_with("f1", 0.82)
        mock_params.assert_called_once_with(params)

        mock_tag.assert_any_call("mlflow.runName", "Pipeline_PCA")
        mock_artifact.assert_called_once_with("caminho/modelo.joblib")

    def test_salvar_resumo_json_persistencia(self, tmp_path):
        tracker = MLflowTracker()
        output_file = tmp_path / "resumo_teste.json"
        metrics = {"f1": 0.85, "best_threshold": 0.5}

        tracker.salvar_resumo_json("Modelo_V1", metrics, output_file)

        assert output_file.exists()
        with open(output_file, "r") as f:
            conteudo = json.load(f)
            assert conteudo["model"] == "Modelo_V1"
            assert conteudo["f1_holdout"] == 0.85

    @patch("src.modeling.tracker.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_salvar_modelo_com_alias_producao(self, mock_log_model, mock_client_class):
        tracker = MLflowTracker()
        mock_model = MagicMock()

        mock_model_info = MagicMock()
        mock_model_info.registered_model_version = "3"
        mock_log_model.return_value = mock_model_info

        tracker.salvar_modelo(mock_model, model_name="CreditRiskModel", artifact_path="model_v3")

        args, kwargs = mock_log_model.call_args
        assert kwargs["name"] == "model_v3"
        assert "skops_trusted_types" in kwargs

        mock_client_instance = mock_client_class.return_value
        mock_client_instance.set_registered_model_alias.assert_called_once_with(
            name="CreditRiskModel",
            alias="production",
            version="3"
        )
