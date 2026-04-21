"""
src/__init__.py — Pacote raiz do Pipeline de Risco de Crédito.

Este ficheiro exporta a API pública do projeto, permitindo importações simplificadas.
Centraliza componentes de Orquestração, Ingestão, Qualidade, Pré-processamento e Modelagem.
"""

from src.core.context import PipelineContext
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester
from src.modeling import FeatureReducer, ModelingStep
from src.preprocessing import (
    PreprocessingStep,
    PreprocessingPipelineBuilder,
    BaseFeatureTransformer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    CategoricalEncoder,
    CreditFeatureSelector,
    GroupWiseImputer,
    ZScoreScaler,
    FeatureInteractionTransformer,
)
from src.quality import (
    GeExpectationResolver,
    GreatExpectationsValidator,
    QualityReportWriter,
    QualityValidator,
)

__all__ = [
    # ── Orquestração e Contexto ──
    "PipelineContext",

    # ── Ingestão de Dados (UCI/Kaggle) ──
    "KaggleDownloader",
    "CsvToParquetIngester",

    # ── Validação de Qualidade (Great Expectations) ──
    "QualityValidator",
    "GeExpectationResolver",
    "GreatExpectationsValidator",
    "QualityReportWriter",

    # ── Pré-processamento e Feature Engineering ──
    "PreprocessingStep",
    "PreprocessingPipelineBuilder",
    "BaseFeatureTransformer",
    "BinaryFlagTransformer",
    "RatioFeatureTransformer",
    "LogTransformer",
    "CategoricalEncoder",
    "CreditFeatureSelector",
    "GroupWiseImputer",
    "ZScoreScaler",
    "FeatureInteractionTransformer",

    # ── Modelagem e MLOps (Classificação) ──
    "FeatureReducer",
    "ModelingStep",
]
