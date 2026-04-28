"""
core/context.py — PipelineContext: The System Maestro.

Responsibilities:
  - Resolution of absolute paths in the project.
  - Centralized loading of configurations (YAML).
  - Instantiation of the unique Logger (Singleton-like).
  - Dispatch of steps via run_step().
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger


class PipelineContext:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.config_dir = self.root_dir / "config"

        self.pipeline_cfg = load_yaml(self.config_dir / "pipeline.yaml")
        self.data_cfg = load_yaml(self.config_dir / "data.yaml")
        self.quality_cfg = load_yaml(self.config_dir / "quality.yaml")

        self.preprocessing_cfg = load_yaml(self.config_dir / "preprocessing.yaml")
        self.modeling_cfg = load_yaml(self.config_dir / "modeling.yaml")

        log_cfg = self.pipeline_cfg.get("logging", {})
        self.logger = get_logger("CreditRiskPipeline", log_cfg)

        paths_cfg = self.pipeline_cfg.get("paths", {})
        self.raw_dir = self.root_dir / paths_cfg.get("raw_data_dir", "data/raw")
        self.processed_dir = self.root_dir / paths_cfg.get("processed_data_dir", "data/processed")
        self.features_dir = self.root_dir / paths_cfg.get("features_data_dir", "data/features")

        self.output_path = self.processed_dir / "credit_risk.parquet"

        for d in [self.raw_dir, self.processed_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_notebook(cls, notebook_path: str) -> PipelineContext:
        root = Path(notebook_path).resolve().parent.parent
        return cls(root)

    def run_step(self, step_name: str) -> None:
        if step_name == "ingestion":
            self._run_ingestion()
        elif step_name == "quality":
            self._run_quality()
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _run_ingestion(self) -> None:
        from src.ingestion.downloader import KaggleDownloader
        from src.ingestion.parquet_writer import CsvToParquetIngester

        kaggle_cfg = self.data_cfg.get("kaggle", {})
        schema_cfg = self.data_cfg.get("schema", {})

        downloader = KaggleDownloader(
            dataset=kaggle_cfg.get("dataset"),
            logger=self.logger
        )
        downloader.load(self.raw_dir)

        ingester = CsvToParquetIngester(
            raw_dir=self.raw_dir,
            output_path=self.output_path,
            compression=self.data_cfg.get("ingest", {}).get("compression", "snappy"),
            validate_schema=self.data_cfg.get("ingest", {}).get("validate_schema", True),
            required_columns=schema_cfg.get("required_columns", []),
            logger=self.logger
        )
        ingester.run()

    def _run_quality(self) -> None:
        import great_expectations as gx
        from src.quality import (
            GreatExpectationsValidator,
            GeExpectationResolver,
            QualityReportWriter
        )

        if not self.output_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.output_path}")

        self.logger.info("Validating Quality: %s", self.output_path.name)
        df = pd.read_parquet(self.output_path)

        full_quality_cfg = {**self.pipeline_cfg, **self.quality_cfg}

        resolver = GeExpectationResolver(gx.expectations)
        validator = GreatExpectationsValidator(resolver, self.logger, gx)
        summary = validator.validate(df, full_quality_cfg)

        quality_meta = self.quality_cfg.get("quality", {})
        output_dir = self.root_dir / quality_meta.get("output_dir", "outputs/quality")

        writer = QualityReportWriter(self.logger)
        writer.write(summary, output_dir)
