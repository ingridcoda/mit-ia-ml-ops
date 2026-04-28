"""
quality/ge_validator.py — Implementation of the Great Expectations-based data quality validator for financial checkpoints.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.quality.base import ExpectationResolver, QualityValidator


class GreatExpectationsValidator(QualityValidator):
    def __init__(self, resolver: ExpectationResolver, logger: logging.Logger, gx) -> None:
        super().__init__(logger)
        self._resolver = resolver
        self._gx = gx

    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        self._logger.info("Starting data quality validation with Great Expectations.")

        context = self._gx.get_context(mode="ephemeral")

        datasource = context.data_sources.add_pandas(name="credit_source")
        asset = datasource.add_dataframe_asset(name="credit_asset")
        batch_def = asset.add_batch_definition_whole_dataframe("credit_batch_def")

        suite_name = config.get("quality", {}).get("suite_name", "credit_suite")
        suite = self._gx.ExpectationSuite(name=suite_name)

        self._add_expectations(suite, config.get("table_expectations", []), config.get("column_expectations", {}))
        suite = context.suites.add(suite)

        validation_def = self._gx.ValidationDefinition(
            name="credit_validation",
            data=batch_def,
            suite=suite
        )
        validation_def = context.validation_definitions.add(validation_def)

        checkpoint = self._gx.Checkpoint(
            name="credit_checkpoint",
            validation_definitions=[validation_def]
        )
        checkpoint = context.checkpoints.add(checkpoint)

        results = checkpoint.run(batch_parameters={"dataframe": df})

        summary = {
            "success": results.success,
            "results_object": results
        }

        if not results.success and config.get("quality", {}).get("fail_pipeline_on_error", False):
            self._logger.error("Data quality validation failed. Failing pipeline as per configuration.")
            raise RuntimeError("Qualidade de dados insuficiente para o treinamento.")

        return summary

    def _add_expectations(self, suite, table_exps, column_exps):
        for exp in table_exps:
            cls = self._resolver.resolve(exp["type"])
            suite.add_expectation(cls(**exp.get("kwargs", {})))

        for col, exps in column_exps.items():
            for exp in exps:
                cls = self._resolver.resolve(exp["type"])
                suite.add_expectation(cls(column=col, **exp.get("kwargs", {})))
