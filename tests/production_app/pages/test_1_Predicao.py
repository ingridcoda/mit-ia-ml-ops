"""
test_1_Predicao.py — Simulação de preenchimento e inferência no Streamlit.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from streamlit.testing.v1 import AppTest

# Calcula a raiz do projeto (3 níveis acima deste arquivo)
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PAGE_PATH = _ROOT / "production_app" / "pages" / "1_Predicao.py"


def test_fluxo_predicao_aprovada():
    """Simula um cliente de baixo risco e verifica a mensagem de aprovação."""
    at = AppTest.from_file(str(_PAGE_PATH))  # Caminho absoluto

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    with patch("utils.model_utils.load_credit_model", return_value=mock_model), \
            patch("pathlib.Path.exists", return_value=True):
        at.run()
        at.number_input[0].set_value(30)  # Idade
        at.number_input[2].set_value(1000)  # Valor
        at.button[0].click().run()

        assert "APROVADO" in at.success[0].value
