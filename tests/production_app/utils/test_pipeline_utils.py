"""
tests/production_app/utils/test_pipeline_utils.py — Tests for data sanitization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_APP_DIR = _ROOT / "production_app"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.pipeline_utils import preprocess_inputs, get_parquet_features


@pytest.fixture
def input_client_base():
    return {
        "Age": 30,
        "Job": 2,
        "Credit amount": 5000,
        "Duration": 12,
        "Checking account": "little"
    }


class TestPipelineUtils:
    def test_preprocess_inputs_happy_path(self, input_client_base):
        df = preprocess_inputs(input_client_base)

        assert len(df) == 1
        assert list(df.columns) == [
            "Age", "Job", "Credit amount", "log_Credit amount",
            "Duration", "checking_acc_encoded", "is_high_risk_amount"
        ]
        assert df["log_Credit amount"].iloc[0] == pytest.approx(8.51, abs=0.01)

    @pytest.mark.parametrize("value,expected", [
        (5000, 0),
        (5001, 1),
        (10000, 1)
    ])
    def test_is_high_risk_amount_threshold(self, input_client_base, value, expected):
        input_client_base["Credit amount"] = value
        df = preprocess_inputs(input_client_base)
        assert df["is_high_risk_amount"].iloc[0] == expected

    @pytest.mark.parametrize("category,encoded_value", [
        ("little", 0),
        ("moderate", 1),
        ("rich", 2),
        ("NA", 3),
        ("unknown", 3)
    ])
    def test_checking_account_encoding(self, input_client_base, category, encoded_value):
        input_client_base["Checking account"] = category
        df = preprocess_inputs(input_client_base)
        assert df["checking_acc_encoded"].iloc[0] == encoded_value

    def test_discard_unused_columns(self, input_client_base):
        input_client_base["Useless_Column"] = "test"
        df = preprocess_inputs(input_client_base)

        assert "Useless_Column" not in df.columns
        assert df.shape[1] == 7

    def test_get_parquet_features_returns_path(self):
        path = get_parquet_features()
        assert isinstance(path, Path)
        assert path.name == "credit_features.parquet"
