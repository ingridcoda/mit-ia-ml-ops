"""
src/__init__.py — Root package of the Credit Risk Pipeline.

This file exports the project's public API, allowing simplified imports.
Centralizes components of Orchestration, Ingestion, Quality, Preprocessing and Modeling.
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

    "PipelineContext",

    "KaggleDownloader",
    "CsvToParquetIngester",

    "QualityValidator",
    "GeExpectationResolver",
    "GreatExpectationsValidator",
    "QualityReportWriter",

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

    "FeatureReducer",
    "ModelingStep",
]
