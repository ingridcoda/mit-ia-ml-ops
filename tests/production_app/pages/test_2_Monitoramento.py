"""test_2_Monitoramento.py — Tests for the monitoring page."""

from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PAGE_PATH = _ROOT / "production_app" / "pages" / "2_Monitoramento.py"


def test_monitoramento_carrega_versoes_dinamicamente():
    at = AppTest.from_file(str(_PAGE_PATH))

    m_v1 = MagicMock(spec=Path)
    m_v1.name = "v1"
    m_v1.is_dir.return_value = True

    m_v1.glob.return_value = [Path("m.json")]

    df_fake = pd.DataFrame({'Age': np.random.rand(1000), 'Credit amount': np.random.rand(1000)})
    json_mock = '{"f1": 0.8, "accuracy": 0.85, "roc_auc": 0.78, "selected_reducer": "pca"}'

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.glob", return_value=[m_v1]), patch(
            "pandas.read_parquet", return_value=df_fake), patch("builtins.open", mock_open(read_data=json_mock)):
        at.run()

        assert not at.exception

        assert len(at.tabs) > 0
