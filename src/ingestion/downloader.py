"""
ingestion/downloader.py — KaggleDownloader

Responsible for extracting raw data from the original source (Kaggle/UCI).
Ensures that the download only occurs if the data is not present (Idempotency).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.core.base import DataLoader


class KaggleDownloader(DataLoader):
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
        self._authenticate()

    def _authenticate(self) -> None:
        load_dotenv()
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            self.logger.error("KAGGLE_USERNAME or KAGGLE_KEY credentials not found in .env")
            raise EnvironmentError("Configure Kaggle credentials before starting.")

    def load(self, destination_dir: Path) -> list[Path]:
        destination_dir.mkdir(parents=True, exist_ok=True)

        if self.skip_if_exists and not self.force and any(destination_dir.iterdir()):
            self.logger.info("Data already present in %s. Skipping download.", destination_dir)
            return list(destination_dir.iterdir())

        self.logger.info("Starting download of the dataset: %s", self.dataset)

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(self.dataset, path=str(destination_dir), unzip=True)
        self.logger.info("Download and extraction completed successfully in %s", destination_dir)

        return list(destination_dir.iterdir())
