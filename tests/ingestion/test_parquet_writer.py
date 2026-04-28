"""
tests/test_parquet_writer.py — Unit tests for the CsvToParquetIngester.

Ensures that the conversion of raw credit data to the performance format (Parquet) occurs without loss of information.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.parquet_writer import CsvToParquetIngester


class TestCsvToParquetIngester:
    def test_conversao_csv_para_parquet_sucesso(self, tmp_raw_dir, tmp_path, null_logger):
        """Valida o fluxo completo de conversão e persistência."""

        csv_path = tmp_raw_dir / "german_credit.csv"
        df_orig = pd.DataFrame({
            "Age": [20, 30],
            "Credit amount": [1000, 2000],
            "Risk": ["good", "bad"]
        })
        df_orig.to_csv(csv_path, index=False)

        output_parquet = tmp_path / "credit_final.parquet"

        ingester = CsvToParquetIngester(
            raw_dir=tmp_raw_dir,
            output_path=output_parquet,
            logger=null_logger,
            required_columns=["Risk"]
        )
        ingester.run()

        assert output_parquet.exists()
        df_lido = pd.read_parquet(output_parquet)
        assert df_lido.shape == (2, 3)
        assert "Risk" in df_lido.columns

    def test_erro_se_coluna_obrigatoria_ausente(self, tmp_raw_dir, tmp_path, null_logger):
        """Garante que o pipeline quebra se o CSV vier sem a coluna de alvo (Risk)."""

        pd.DataFrame({"Age": [20]}).to_csv(tmp_raw_dir / "bad_data.csv", index=False)

        output_parquet = tmp_path / "fail.parquet"
        ingester = CsvToParquetIngester(
            raw_dir=tmp_raw_dir,
            output_path=output_parquet,
            logger=null_logger,
            required_columns=["Risk"],
            validate_schema=True
        )

        with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
            ingester.run()

    def test_compressao_snappy(self, tmp_raw_dir, tmp_path, null_logger):
        """Verifica se o arquivo Parquet é gerado com compressão Snappy (requisito de performance)."""
        pd.DataFrame({"A": [1]}).to_csv(tmp_raw_dir / "data.csv", index=False)
        output = tmp_path / "compressed.parquet"

        ingester = CsvToParquetIngester(tmp_raw_dir, output, null_logger, compression="snappy")
        ingester.run()

        metadata = pq.read_metadata(output)

        assert metadata.row_group(0).column(0).compression == "SNAPPY"

    def test_erro_se_diretorio_vazio(self, tmp_path, null_logger):
        """Garante que o ingester levanta erro claro se não encontrar CSVs para processar."""
        dir_vazio = tmp_path / "vazio"
        dir_vazio.mkdir()
        output_parquet = tmp_path / "nada.parquet"

        ingester = CsvToParquetIngester(
            raw_dir=dir_vazio,
            output_path=output_parquet,
            logger=null_logger
        )

        with pytest.raises(FileNotFoundError, match="Nenhum CSV encontrado"):
            ingester.run()
