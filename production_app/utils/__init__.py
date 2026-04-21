"""
utils/__init__.py — Exportação dos utilitários de produção.
"""
from .model_utils import load_credit_model
from .pipeline_utils import preprocessar_entradas, obter_parquet_features

__all__ = [
    "preprocessar_entradas",
    "obter_parquet_features",
    "load_credit_model"
]
