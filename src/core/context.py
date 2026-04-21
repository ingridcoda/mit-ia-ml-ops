"""
core/context.py — PipelineContext: O Maestro do Sistema.

Responsabilidades:
  - Resolução de caminhos absolutos no projeto.
  - Carregamento centralizado de configurações (YAML).
  - Instanciação do Logger único (Singleton-like).
  - Despacho de etapas via run_step().
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger


class PipelineContext:
    """
    Contexto de execução que centraliza recursos para todas as etapas.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.config_dir = self.root_dir / "config"

        # Carregamento dos Contratos (YAML)
        self.pipeline_cfg = load_yaml(self.config_dir / "pipeline.yaml")
        self.data_cfg = load_yaml(self.config_dir / "data.yaml")
        self.quality_cfg = load_yaml(self.config_dir / "quality.yaml")

        # Ajuste: Adicionando as configurações de pré-processamento e modelagem
        self.preprocessing_cfg = load_yaml(self.config_dir / "preprocessing.yaml")
        self.modeling_cfg = load_yaml(self.config_dir / "modeling.yaml")

        # Configuração de Infra
        log_cfg = self.pipeline_cfg.get("logging", {})
        self.logger = get_logger("CreditRiskPipeline", log_cfg)

        # Resolução de Caminhos de Dados
        paths_cfg = self.pipeline_cfg.get("paths", {})
        self.raw_dir = self.root_dir / paths_cfg.get("raw_data_dir", "data/raw")
        self.processed_dir = self.root_dir / paths_cfg.get("processed_data_dir", "data/processed")
        self.features_dir = self.root_dir / paths_cfg.get("features_data_dir", "data/features")

        # Nome do arquivo final de saída da ingestão
        self.output_path = self.processed_dir / "credit_risk.parquet"

        # Garante que os diretórios existam
        for d in [self.raw_dir, self.processed_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_notebook(cls, notebook_path: str) -> PipelineContext:
        """Helper para instanciar o contexto a partir de scripts em notebooks/."""
        root = Path(notebook_path).resolve().parent.parent
        return cls(root)

    def run_step(self, step_name: str) -> None:
        """Despachante de etapas do pipeline."""
        if step_name == "ingestion":
            self._run_ingestion()
        elif step_name == "quality":
            self._run_quality()
        else:
            raise ValueError(f"Etapa desconhecida: {step_name}")

    def _run_ingestion(self) -> None:
        """Executa a carga e conversão inicial dos dados."""
        from src.ingestion.downloader import KaggleDownloader
        from src.ingestion.parquet_writer import CsvToParquetIngester

        kaggle_cfg = self.data_cfg.get("kaggle", {})
        schema_cfg = self.data_cfg.get("schema", {})

        # 1. Download
        downloader = KaggleDownloader(
            dataset=kaggle_cfg.get("dataset"),
            logger=self.logger
        )
        downloader.load(self.raw_dir)

        # 2. Conversão Parquet
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
        """Executa a validação de qualidade via Great Expectations."""
        import great_expectations as gx
        from src.quality import (
            GreatExpectationsValidator,
            GeExpectationResolver,
            QualityReportWriter
        )

        if not self.output_path.exists():
            raise FileNotFoundError(f"Parquet não encontrado: {self.output_path}")

        self.logger.info("Validando Qualidade: %s", self.output_path.name)
        df = pd.read_parquet(self.output_path)

        # Merge de configs para o validador
        full_quality_cfg = {**self.pipeline_cfg, **self.quality_cfg}

        # Correção aqui: passando gx.expectations
        resolver = GeExpectationResolver(gx.expectations)
        validator = GreatExpectationsValidator(resolver, self.logger, gx)
        summary = validator.validate(df, full_quality_cfg)

        # Escrita do Relatório
        quality_meta = self.quality_cfg.get("quality", {})
        output_dir = self.root_dir / quality_meta.get("output_dir", "outputs/quality")

        writer = QualityReportWriter(self.logger)
        writer.write(summary, output_dir)
