"""
ingestion/downloader.py — KaggleDownloader

Responsável por extrair os dados brutos da fonte original (Kaggle/UCI).
Garante que o download só ocorra se os dados não estiverem presentes (Idempotência).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.base import DataLoader


class KaggleDownloader(DataLoader):
    """
    Realiza o download do dataset de Risco de Crédito.

    Implementa segurança de ambiente: as chaves de API devem estar no arquivo .env
    para evitar vazamento de credenciais no código.
    """

    def __init__(
            self,
            dataset: str,
            logger: logging.Logger,
            skip_if_exists: bool = True,
            force: bool = False,
    ) -> None:
        super().__init__(logger=logger)
        self.dataset = dataset
        self.skip_if_exists = skip_if_exists
        self.force = force
        self._autenticar()

    def _autenticar(self) -> None:
        """Valida credenciais do Kaggle via variáveis de ambiente."""
        load_dotenv()
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            self.logger.error("Credenciais KAGGLE_USERNAME ou KAGGLE_KEY não encontradas no .env")
            raise EnvironmentError("Configure as credenciais do Kaggle antes de iniciar.")

    def load(self, destination_dir: Path) -> list[Path]:
        """Faz o download e extrai o conteúdo do dataset."""
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Se os arquivos já existem e não forçarmos o download, pulamos
        if self.skip_if_exists and not self.force and any(destination_dir.iterdir()):
            self.logger.info("Dados já presentes em %s. Pulando download.", destination_dir)
            return list(destination_dir.iterdir())

        self.logger.info("Iniciando download do dataset: %s", self.dataset)

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(self.dataset, path=str(destination_dir), unzip=True)
        self.logger.info("Download e extração concluídos com sucesso em %s", destination_dir)

        return list(destination_dir.iterdir())
