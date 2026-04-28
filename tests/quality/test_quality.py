"""
test_quality.py — Unit tests for the data auditing engine.
"""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.quality.expectation_resolver import GeExpectationResolver
from src.quality.ge_validator import GreatExpectationsValidator
from src.quality.report_writer import QualityReportWriter


def test_resolver_converte_snake_case_para_ge():
    gxe_mock = MagicMock()
    gxe_mock.ExpectColumnValuesToBeBetween = "ClasseMockada"

    resolver = GeExpectationResolver(gxe_mock)
    resultado = resolver.resolve("expect_column_values_to_be_between")

    assert resultado == "ClasseMockada"


def test_resolver_lanca_erro_se_regra_invalida():
    gxe_mock = MagicMock()
    del gxe_mock.RegraInexistente
    del gxe_mock.regra_inexistente

    resolver = GeExpectationResolver(gxe_mock)
    with pytest.raises(AttributeError, match="não suportada pelo motor GE"):
        resolver.resolve("regra_inexistente")


def test_quality_report_writer_cria_json(tmp_path, null_logger):
    writer = QualityReportWriter(null_logger)
    summary = {"success": True}

    report_path = writer.write(summary, tmp_path)

    assert report_path.exists()
    assert report_path.suffix == ".json"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["success"] is True
        assert "timestamp" in data
        assert "Auditoria concluída" in data["message"]


def test_ge_validator_sucesso(null_logger):
    gx_mock = MagicMock()
    context_mock = gx_mock.get_context.return_value
    results_mock = MagicMock()
    results_mock.success = True
    context_mock.checkpoints.add.return_value.run.return_value = results_mock

    resolver_mock = MagicMock()
    validator = GreatExpectationsValidator(resolver_mock, null_logger, gx_mock)

    df = pd.DataFrame({"A": [1, 2, 3]})
    config = {
        "quality": {"suite_name": "test_suite", "fail_pipeline_on_error": True},
        "table_expectations": [],
        "column_expectations": {}
    }

    summary = validator.validate(df, config)
    assert summary["success"] is True


def test_ge_validator_fail_fast_quebra_pipeline(null_logger):
    gx_mock = MagicMock()
    context_mock = gx_mock.get_context.return_value
    results_mock = MagicMock()
    results_mock.success = False
    context_mock.checkpoints.add.return_value.run.return_value = results_mock

    resolver_mock = MagicMock()
    validator = GreatExpectationsValidator(resolver_mock, null_logger, gx_mock)

    df = pd.DataFrame({"A": [1, 2, 3]})
    config = {
        "quality": {"fail_pipeline_on_error": True}
    }

    with pytest.raises(RuntimeError, match="Qualidade de dados insuficiente"):
        validator.validate(df, config)
