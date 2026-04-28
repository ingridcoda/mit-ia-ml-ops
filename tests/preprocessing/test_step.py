"""
tests/test_step.py — Integration test for PreprocessingStep.

Validates if the preprocessing step correctly orchestrates the
custom credit transformers.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.preprocessing.step import PreprocessingStep


def _criar_context_mock(tmp_path, df, logger):
    """Simulates the PipelineContext for the integration test."""
    ctx = MagicMock()
    ctx.logger = logger
    ctx.root_dir = tmp_path
    ctx.output_path = tmp_path / "raw_credit.parquet"

    ctx.features_dir = tmp_path

    df.to_parquet(ctx.output_path, index=False)

    ctx.preprocessing_cfg = {
        "preprocessing": {"output_filename": "credit_features.parquet"},
        "binary_flags": [{"column": "Age", "value": 60, "new_column": "is_senior"}],
        "ratio_features": [{"name": "installment", "numerator": "Credit amount", "denominator": "Duration"}],
        "log_transform": {"columns": ["Credit amount"]},
        "feature_selection": {"features_to_keep": ["is_senior", "installment", "log_Credit amount", "Risk"]}
    }
    return ctx


class TestPreprocessingStepIntegration:
    def test_execucao_completa_pipeline_credito(self, tmp_path, sample_credit_df, null_logger):
        """Ensures that the Step generates the features file with the transformed columns."""
        ctx = _criar_context_mock(tmp_path, sample_credit_df, null_logger)
        step = PreprocessingStep(ctx)

        step.run()

        caminho_saida = ctx.features_dir / "credit_features.parquet"

        assert caminho_saida.exists()

        df_out = pd.read_parquet(caminho_saida)

        colunas_esperadas = ["is_senior", "installment", "log_Credit amount", "Risk"]
        for col in colunas_esperadas:
            assert col in df_out.columns, f"Feature {col} was not generated in the pipeline."

        assert df_out["log_Credit amount"].iloc[0] == pytest.approx(6.908, abs=0.1)
