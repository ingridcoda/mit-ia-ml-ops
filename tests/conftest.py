"""conftest.py — Pytest configuration and fixtures."""

import logging
from pathlib import Path

import mlflow
import pandas as pd
import pytest


def pytest_configure(config):
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

    logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.ERROR)


@pytest.fixture(autouse=True, scope="session")
def setup_mlflow_test_env():
    mlflow.set_tracking_uri("sqlite:///:memory:")
    yield


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_credit_df() -> pd.DataFrame:
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
    logger = logging.getLogger("test_null")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
