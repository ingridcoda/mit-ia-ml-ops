"""
src/core/__init__.py — Exporta a API principal do núcleo do pipeline.

Ao expor PipelineContext e PipelineStep aqui, permitimos importações como:
    from src.core import PipelineContext
Em vez de:
    from src.core.context import PipelineContext
"""

from src.core.base import PipelineStep, DataLoader
from src.core.context import PipelineContext

__all__ = [
    "PipelineStep",
    "DataLoader",
    "PipelineContext"
]
