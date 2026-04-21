"""
test_2_Monitoramento.py — Validação visual do painel de monitoramento.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Calcula a raiz do projeto para localizar o script
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PAGE_PATH = _ROOT / "production_app" / "pages" / "2_Monitoramento.py"


def test_monitoramento_exibe_kpis_e_drift():
    """Garante que as métricas e a tabela de drift estão na tela."""
    at = AppTest.from_file(str(_PAGE_PATH))

    df_fake = pd.DataFrame({
        'Age': [25, 30],
        'Duration': [12, 24],
        'Credit amount': [1000, 2000],
        'log_Credit amount': [np.log1p(1000), np.log1p(2000)],  # Coluna faltante corrigida
        'Risk': ['good', 'bad']
    })

    with patch("pandas.read_parquet", return_value=df_fake), \
            patch("pathlib.Path.exists", return_value=True):
        at.run()

        # Verifica se os subheaders principais estão presentes
        assert "Performance do Modelo" in at.subheader[0].value
        assert "Detecção de Drift" in at.subheader[1].value

        # Verifica se não houve erro de execução na árvore de elementos
        assert not at.exception
