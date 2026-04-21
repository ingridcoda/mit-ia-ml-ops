"""
src/preprocessing/__init__.py — API pública de Pré-processamento.
"""
from src.preprocessing.base import BaseFeatureTransformer
from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from src.preprocessing.step import PreprocessingStep
from src.preprocessing.transformers import (
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    CategoricalEncoder,
    CreditFeatureSelector,
    GroupWiseImputer,
    ZScoreScaler,
    FeatureInteractionTransformer
)

__all__ = [
    "BaseFeatureTransformer",
    "PreprocessingPipelineBuilder",
    "PreprocessingStep",
    "BinaryFlagTransformer",
    "RatioFeatureTransformer",
    "LogTransformer",
    "CategoricalEncoder",
    "CreditFeatureSelector",
    "GroupWiseImputer",
    "ZScoreScaler",
    "FeatureInteractionTransformer",
]
