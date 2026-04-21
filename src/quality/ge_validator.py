"""
quality/ge_validator.py — Implementação Great Expectations para Crédito (API V1.0+).
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
        self._logger.info("Iniciando auditoria de qualidade (Checkpoints Financeiros)...")

        context = self._gx.get_context(mode="ephemeral")

        # 1. Configuração de Dados
        datasource = context.data_sources.add_pandas(name="credit_source")
        asset = datasource.add_dataframe_asset(name="credit_asset")
        batch_def = asset.add_batch_definition_whole_dataframe("credit_batch_def")

        # 2. Configuração de Regras (Suite) - Nova Sintaxe
        suite_name = config.get("quality", {}).get("suite_name", "credit_suite")
        suite = self._gx.ExpectationSuite(name=suite_name)

        self._add_expectations(suite, config.get("table_expectations", []), config.get("column_expectations", {}))
        suite = context.suites.add(suite)  # Adiciona a suite via manager

        # 3. Definição de Validação (O elo entre Dados e Regras na V1.0)
        validation_def = self._gx.ValidationDefinition(
            name="credit_validation",
            data=batch_def,
            suite=suite
        )
        validation_def = context.validation_definitions.add(validation_def)

        # 4. Checkpoint - Nova Sintaxe
        checkpoint = self._gx.Checkpoint(
            name="credit_checkpoint",
            validation_definitions=[validation_def]
        )
        checkpoint = context.checkpoints.add(checkpoint)

        # 5. Execução (Aqui o DataFrame entra em cena e a magia acontece)
        results = checkpoint.run(batch_parameters={"dataframe": df})

        summary = {
            "success": results.success,
            "results_object": results
        }

        # Lógica de Fail-Fast de Engenharia
        if not results.success and config.get("quality", {}).get("fail_pipeline_on_error", False):
            self._logger.error("DADOS DE CRÉDITO REPROVADOS NA AUDITORIA. Interrompendo pipeline.")
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
