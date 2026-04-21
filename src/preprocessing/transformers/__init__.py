"""
src/preprocessing/transformers/__init__.py — API pública dos transformadores de crédito.

Centraliza a exportação de todos os componentes de engenharia de features,
garantindo que o PipelineBuilder e os notebooks possam acessar aos transformadores
de forma organizada e padronizada.
"""

from src.preprocessing.transformers.binary_flags import BinaryFlagTransformer
from src.preprocessing.transformers.categorical_encoder import CategoricalEncoder
from src.preprocessing.transformers.feature_selector import CreditFeatureSelector
from src.preprocessing.transformers.log_transform import LogTransformer
from src.preprocessing.transformers.polynomial_features import FeatureInteractionTransformer
from src.preprocessing.transformers.ratio_features import RatioFeatureTransformer
from src.preprocessing.transformers.stateful import GroupWiseImputer, ZScoreScaler

__all__ = [
    # Transformadores Stateless
    "BinaryFlagTransformer",
    "RatioFeatureTransformer",
    "LogTransformer",
    "CategoricalEncoder",
    "CreditFeatureSelector",
    "FeatureInteractionTransformer",

    # Transformadores Stateful (aprendem com o treino)
    "GroupWiseImputer",
    "ZScoreScaler",
]
