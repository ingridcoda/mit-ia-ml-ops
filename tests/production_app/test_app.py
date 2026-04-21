"""
test_app.py — Testes de interface para a página inicial (Home).
"""
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

# Calcula a raiz do projeto (2 níveis acima deste arquivo)
_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_PATH = _ROOT / "production_app" / "app.py"


def test_home_status_indicadores_sucesso():
    """Verifica se o status aparece como 'Pronto' quando os arquivos existem."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=30)  # Caminho absoluto

    with patch("pathlib.Path.exists", return_value=True):
        at.run()
        assert "**Modelo de IA**: Pronto" in at.success[0].value
        assert "**Base de Dados**: Conectada" in at.success[1].value


def test_home_status_indicadores_erro():
    """Verifica se o app avisa quando o modelo está faltando."""
    at = AppTest.from_file(str(_APP_PATH))

    with patch("pathlib.Path.exists", return_value=False):
        at.run()
        assert "**Modelo de IA**: Não encontrado" in at.error[0].value
