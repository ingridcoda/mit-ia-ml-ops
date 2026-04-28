"""
ingestion/parquet_writer.py — CsvToParquetIngester

Reads CSV files and converts them to Parquet using PyArrow.
Optimized for backend: uses snappy compression for balance between speed and size.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.core.base import PipelineStep


class CsvToParquetIngester(PipelineStep):
    def __init__(
            self,
            raw_dir: Path,
            output_path: Path,
            logger: logging.Logger,
            required_columns: list[str] | None = None,
            compression: str = "snappy",
            validate_schema: bool = True
    ) -> None:
        super().__init__(logger=logger)
        self.raw_dir = raw_dir
        self.output_path = output_path
        self.required_columns = required_columns or []
        self.compression = compression
        self.validate_schema = validate_schema

    def run(self) -> None:
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {self.raw_dir}")

        self.logger.info("Convertendo %d arquivo(s) CSV para Parquet...", len(csv_files))

        dfs = [pd.read_csv(f) for f in csv_files]
        df_final = pd.concat(dfs, ignore_index=True)

        if self.validate_schema:
            ausentes = [c for c in self.required_columns if c not in df_final.columns]
            if ausentes:
                raise ValueError(f"Colunas obrigatórias ausentes no CSV: {ausentes}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(self.output_path, compression=self.compression, index=False)

        self.logger.info("Ingestão concluída: %s | Shape: %s", self.output_path.name, df_final.shape)
