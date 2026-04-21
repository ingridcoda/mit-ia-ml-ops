"""
conftest.py — Fixtures globais para a suíte de testes de Crédito.
"""
import logging
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    """Diretório temporário para simular data/raw."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def sample_credit_df() -> pd.DataFrame:
    """Gera um DataFrame fake com o schema da UCI Credit."""
    return pd.DataFrame({
        "Age": [25, 40, 60],
        "Sex": ["male", "female", "male"],
        "Job": [2, 1, 3],
        "Credit amount": [1000, 5000, 12000],
        "Duration": [6, 24, 48],
        "Risk": ["good", "bad", "good"]
    })


@pytest.fixture
def null_logger() -> logging.Logger:
    """Logger silencioso para os testes."""
    logger = logging.getLogger("test_null")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
