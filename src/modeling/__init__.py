"""
src/modeling/__init__.py — API pública do módulo de modelagem.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.modeling.reducer import FeatureReducer

if TYPE_CHECKING:
    from src.modeling.step import ModelingStep

__all__ = ["FeatureReducer", "ModelingStep"]


def __getattr__(name: str):
    if name == "ModelingStep":
        from src.modeling.step import ModelingStep as _MS
        return _MS
    raise AttributeError(f"module 'src.modeling' has no attribute {name!r}")
