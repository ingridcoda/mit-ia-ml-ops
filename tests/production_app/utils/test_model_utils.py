"""
tests/production_app/utils/test_model_utils.py — Testes dos utilitários de inferência.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils import load_credit_model

# Bootstrap de caminhos para encontrar o código da app
_TESTS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _TESTS_DIR.parent
_APP_DIR = _PROJECT_ROOT / "production_app"

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TestModelUtils:
    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_credit_model_sucesso(self, mock_exists, mock_load):
        """Verifica se o modelo é carregado corretamente quando o arquivo existe."""
        mock_exists.return_value = True
        mock_load.return_value = MagicMock()

        model = load_credit_model()

        assert model is not None
        mock_load.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_credit_model_arquivo_inexistente(self, mock_exists):
        """Garante que a função retorna None se o arquivo .joblib não for encontrado."""
        mock_exists.return_value = False

        model = load_credit_model()

        assert model is None
