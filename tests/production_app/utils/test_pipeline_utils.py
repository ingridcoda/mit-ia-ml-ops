"""
tests/production_app/utils/test_pipeline_utils.py — Testes da sanitização de dados.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Bootstrap de caminhos
_TESTS_DIR = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _TESTS_DIR.parent
_APP_DIR = _PROJECT_ROOT / "production_app"

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import preprocessar_entradas, obter_parquet_features


@pytest.fixture
def input_cliente_base():
    """Massa de dados base para os testes de inferência."""
    return {
        "Age": 30,
        "Job": 2,
        "Credit amount": 5000,
        "Duration": 12,
        "Checking account": "little"
    }


class TestPipelineUtils:
    def test_preprocessar_entradas_fluxo_feliz(self, input_cliente_base):
        """Verifica se o log e a ordem das colunas estão corretos."""
        df = preprocessar_entradas(input_cliente_base)

        assert len(df) == 1
        assert list(df.columns) == [
            "Age", "Job", "Credit amount", "log_Credit amount",
            "Duration", "checking_acc_encoded", "is_high_risk_amount"
        ]
        # log1p(5000) ~ 8.51
        assert df["log_Credit amount"].iloc[0] == pytest.approx(8.51, abs=0.01)

    @pytest.mark.parametrize("valor,esperado", [
        (5000, 0),  # Limite inferior: não é alto risco
        (5001, 1),  # Acima de 5000: alto risco
        (10000, 1)
    ])
    def test_is_high_risk_amount_threshold(self, input_cliente_base, valor, esperado):
        """Valida a lógica binária do flag de valor de crédito."""
        input_cliente_base["Credit amount"] = valor
        df = preprocessar_entradas(input_cliente_base)
        assert df["is_high_risk_amount"].iloc[0] == esperado

    @pytest.mark.parametrize("categoria,valor_encoded", [
        ("little", 0),
        ("moderate", 1),
        ("rich", 2),
        ("NA", 3),
        ("desconhecido", 3)  # Fallback para nulos/desconhecidos
    ])
    def test_checking_account_encoding(self, input_cliente_base, categoria, valor_encoded):
        """Garante que o mapeamento ordinal da conta corrente está correto."""
        input_cliente_base["Checking account"] = categoria
        df = preprocessar_entradas(input_cliente_base)
        assert df["checking_acc_encoded"].iloc[0] == valor_encoded

    def test_descarte_de_colunas_nao_utilizadas(self, input_cliente_base):
        """Garante que o modelo não receba lixo (colunas extras) do formulário."""
        input_cliente_base["Coluna_Inutil"] = "teste"
        df = preprocessar_entradas(input_cliente_base)

        assert "Coluna_Inutil" not in df.columns
        assert df.shape[1] == 7  # Número exato de features esperadas pelo XGBoost

    def test_obter_parquet_features_retorno_path(self):
        """Verifica se a função retorna o objeto Path correto para o monitoramento."""
        path = obter_parquet_features()
        assert isinstance(path, Path)
        assert path.name == "credit_features.parquet"
