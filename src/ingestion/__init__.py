"""
src/ingestion/__init__.py — API pública do módulo de ingestão.
"""
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester

__all__ = [
    "KaggleDownloader",
    "CsvToParquetIngester"
]
