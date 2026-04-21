"""
quality/base.py — Contratos para Validação de Dados de Crédito.

Define as interfaces para:
  - QualityValidator    : O motor de execução dos checks.
  - ExpectationResolver : O tradutor de regras (YAML -> Código).
  - QualityReportWriter : O gerador de evidências para auditoria.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class QualityValidator(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        """Executa a bateria de testes de qualidade definida no config."""


class ExpectationResolver(ABC):
    @abstractmethod
    def resolve(self, type_name: str) -> type:
        """Resolve nomes de strings do YAML para classes executáveis."""


class QualityReportWriterBase(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        """Persiste os resultados da validação em um artefato técnico."""
