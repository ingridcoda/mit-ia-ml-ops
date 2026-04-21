"""
ingestion/parquet_writer.py — CsvToParquetIngester

Lê os arquivos CSV e os converte para Parquet usando PyArrow.
Otimizado para backend: usa compressão snappy para balanço entre velocidade e tamanho.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.core.base import PipelineStep


class CsvToParquetIngester(PipelineStep):
    """
    Consolida arquivos CSV em um único Parquet sanitizado.
    """

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
        """Executa a conversão e valida o schema de saída."""
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {self.raw_dir}")

        self.logger.info("Convertendo %d arquivo(s) CSV para Parquet...", len(csv_files))

        # Leitura e concatenação (para datasets médios como o UCI Credit)
        dfs = [pd.read_csv(f) for f in csv_files]
        df_final = pd.concat(dfs, ignore_index=True)

        # Validação de sanidade inicial
        if self.validate_schema:
            ausentes = [c for c in self.required_columns if c not in df_final.columns]
            if ausentes:
                raise ValueError(f"Colunas obrigatórias ausentes no CSV: {ausentes}")

        # Escrita em Parquet
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(self.output_path, compression=self.compression, index=False)

        self.logger.info("Ingestão concluída: %s | Shape: %s", self.output_path.name, df_final.shape)
