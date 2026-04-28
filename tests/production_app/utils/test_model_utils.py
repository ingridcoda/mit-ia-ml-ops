"""
tests/production_app/utils/test_model_utils.py — Validation of loading utilities.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_APP_DIR = _ROOT / "production_app"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(1, str(_ROOT))

from utils.model_utils import load_credit_model


class TestModelUtils:
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.set_tracking_uri")
    def test_load_credit_model_via_mlflow(self, mock_uri, mock_load):
        mock_load.return_value = MagicMock()
        modelo = load_credit_model()

        mock_uri.assert_called_once_with("sqlite:///mlruns.db")
        mock_load.assert_called_once_with("models:/CreditRiskModel@production")
        assert modelo is not None

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.set_tracking_uri")
    def test_load_credit_model_fallback_joblib(self, mock_uri, mock_mlflow_load, mock_exists, mock_joblib_load):
        mock_mlflow_load.side_effect = Exception("MLflow Error")
        mock_exists.return_value = True
        mock_joblib_load.return_value = MagicMock()

        modelo = load_credit_model()

        mock_joblib_load.assert_called_once()
        assert modelo is not None

    @patch("pathlib.Path.exists")
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.set_tracking_uri")
    def test_load_credit_model_inexistente(self, mock_uri, mock_mlflow_load, mock_exists):
        mock_mlflow_load.side_effect = Exception("No MLflow")
        mock_exists.return_value = False

        modelo = load_credit_model()
        assert modelo is None
