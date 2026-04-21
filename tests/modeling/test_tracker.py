"""
test_tracker.py — Suíte completa de testes para o rastreador de experiências.
"""
import json
from unittest.mock import patch

from src.modeling.tracker import MLflowTracker


class TestMLflowTracker:
    @patch("mlflow.log_artifact")
    @patch("mlflow.set_tag")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metric")
    def test_log_experimento_fluxo_completo(self, mock_metric, mock_params, mock_tag, mock_artifact):
        """Garante que métricas numéricas, parâmetros e tags são registados corretamente."""
        tracker = MLflowTracker()
        # 'status' é string e deve ser filtrado; 'f1' é float e deve ser registado
        metrics = {"f1": 0.82, "status": "sucesso"}
        params = {"n_estimators": 100}
        artifacts = {"modelo": "caminho/modelo.joblib"}

        tracker.log_experimento("XGBoost", metrics, params, artifacts)

        # Valida filtro de tipos: f1 entra, status (string) não
        mock_metric.assert_called_once_with("f1", 0.82)
        mock_params.assert_called_once_with(params)
        mock_tag.assert_called_once_with("model_type", "XGBoost")
        mock_artifact.assert_called_once_with("caminho/modelo.joblib")

    def test_salvar_resumo_json_persistencia(self, tmp_path):
        """Verifica se o resumo JSON é gravado corretamente no disco."""
        tracker = MLflowTracker()
        output_file = tmp_path / "resumo_teste.json"
        metrics = {"f1": 0.85, "best_threshold": 0.5}

        tracker.salvar_resumo_json("Modelo_V1", metrics, output_file)

        # Verifica se o ficheiro existe e se o conteúdo bate com a lógica
        assert output_file.exists()
        with open(output_file, "r") as f:
            data = json.load(f)
            assert data["model"] == "Modelo_V1"
            assert data["f1_holdout"] == 0.85
            assert data["threshold"] == 0.5
