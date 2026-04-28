"""
src/ingestion/__init__.py — Public API of the ingestion module.
"""
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester

__all__ = [
    "KaggleDownloader",
    "CsvToParquetIngester"
]
