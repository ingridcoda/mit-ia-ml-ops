"""
test_context.py — Testes de orquestração e resolução de caminhos.
"""
from unittest.mock import patch

import pytest

from src.core.context import PipelineContext


@pytest.fixture
def mock_configs():
    """Simula o retorno dos arquivos YAML de configuração."""
    return {
        "pipeline": {
            "logging": {"level": "INFO"},
            "paths": {"raw_data_dir": "raw", "features_data_dir": "feat"}
        },
        "data": {"kaggle": {"dataset": "test/data"}},
        "quality": {"quality": {"output_dir": "quality_out"}},
        "preprocessing": {},
        "modeling": {"target_column": "Risk"}
    }


@patch("src.core.context.load_yaml")
@patch("src.core.context.get_logger")
def test_context_inicializacao_e_caminhos(mock_logger, mock_load, tmp_path, mock_configs):
    """Valida se o contexto resolve caminhos e cria diretórios corretamente."""
    # Configuramos o mock para retornar os dicionários conforme o Context espera
    mock_load.side_effect = [
        mock_configs["pipeline"],
        mock_configs["data"],
        mock_configs["quality"],
        mock_configs["preprocessing"],
        mock_configs["modeling"]
    ]

    ctx = PipelineContext(tmp_path)

    # Verifica resolução de caminhos baseada no YAML
    assert ctx.raw_dir == tmp_path / "raw"
    assert ctx.features_dir == tmp_path / "feat"
    assert ctx.output_path == tmp_path / "data/processed" / "credit_risk.parquet"

    # Verifica se o contexto criou as pastas fisicamente no tmp_path
    assert ctx.raw_dir.exists()
    assert ctx.features_dir.exists()


@patch("src.core.context.load_yaml")
def test_context_run_step_erro_desconhecido(mock_load, tmp_path):
    """Garante erro ao tentar rodar uma etapa inexistente."""
    mock_load.return_value = {}
    ctx = PipelineContext(tmp_path)

    with pytest.raises(ValueError, match="Etapa desconhecida"):
        ctx.run_step("etapa_fantasma")


def test_context_from_notebook(tmp_path):
    """Valida o helper de resolução de raiz para notebooks."""
    # Simula a estrutura: root/notebooks/projeto.ipynb
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    fake_ipynb = notebook_dir / "test.ipynb"

    with patch("src.core.context.load_yaml", return_value={}), \
            patch("src.core.context.get_logger"):
        ctx = PipelineContext.from_notebook(str(fake_ipynb))
        # O root deve subir dois níveis a partir do notebook
        assert ctx.root_dir == tmp_path
