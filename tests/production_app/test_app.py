"""test_app.py — Tests for the main app."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_PATH = _ROOT / "production_app" / "app.py"


def test_home_status_indicadores_sucesso():
    at = AppTest.from_file(str(_APP_PATH))
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.glob", return_value=[MagicMock()]), patch(
            "mlflow.search_experiments", return_value=[]):
        at.run()
        assert any("Modelagem" in s.value for s in at.success)


def test_home_status_indicadores_erro():
    at = AppTest.from_file(str(_APP_PATH))
    with patch("pathlib.Path.exists", return_value=False), patch("pathlib.Path.glob", return_value=[]):
        at.run()
        assert any("Modelagem" in e.value for e in at.error)
