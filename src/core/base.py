"""
core/base.py — Classes abstratas base (Contratos) do Pipeline de Crédito.

Define as interfaces fundamentais:
  - PipelineStep : Contrato para etapas do pipeline (Ingestão, Qualidade, Modelagem).
  - DataLoader   : Contrato para carregamento de dados (Kaggle, APIs, SQL).

Este código segue o princípio da Inversão de Dependência (SOLID).
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path


class PipelineStep(ABC):
    """
    Contrato para uma etapa idempotente do pipeline de ML.
    As subclasses recebem o logger compartilhado via PipelineContext.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        """Executa a lógica da etapa. Deve ser idempotente."""


class DataLoader(ABC):
    """
    Contrato para carregamento de dados brutos para o ambiente local.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def load(self, destination_dir: Path) -> list[Path]:
        """
        Busca os dados e os salva no diretório de destino.
        Returns: Lista de caminhos dos arquivos baixados.
        """
