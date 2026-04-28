"""test_1_Predicao.py — Tests for the prediction page."""

from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PAGE_PATH = _ROOT / "production_app" / "pages" / "1_Predicao.py"


class TestPredicaoInterface:
    def setup_method(self):
        self.m_v1 = MagicMock()
        self.m_v1.name = "v1"
        self.m_v1.is_dir.return_value = True
        self.m_v1.glob.side_effect = lambda p: [Path("m.json")] if ".json" in p else [Path("m.joblib")]
        self.m_json = '{"best_threshold": 0.5, "f1": 0.85, "selected_reducer": "pca"}'

    def test_fluxo_predicao_aprovada(self):
        at = AppTest.from_file(str(_PAGE_PATH))
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.iterdir",
                                                                    return_value=[self.m_v1]), patch("builtins.open",
                                                                                                     mock_open(
                                                                                                         read_data=self.m_json)), patch(
            "joblib.load", return_value=mock_model), patch("utils.pipeline_utils.preprocess_inputs",
                                                           return_value=pd.DataFrame(np.zeros((1, 8)))):
            at.run()
            at.button[0].click().run()
            assert any("APROVADO" in s.value for s in at.success)

    def test_geracao_historico_global(self):
        at = AppTest.from_file(str(_PAGE_PATH))
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.iterdir",
                                                                    return_value=[self.m_v1]), patch("builtins.open",
                                                                                                     mock_open(
                                                                                                         read_data=self.m_json)), patch(
            "joblib.load", return_value=mock_model), patch("utils.pipeline_utils.preprocess_inputs",
                                                           return_value=pd.DataFrame(np.zeros((1, 8)))):
            at.run()
            at.button[0].click().run()
            assert len(at.table) > 0
