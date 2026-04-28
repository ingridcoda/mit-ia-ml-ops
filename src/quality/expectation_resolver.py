"""
quality/expectation_resolver.py — Implementation of the ExpectationResolver for mapping quality rule names to Great Expectations expectation classes.
"""
from __future__ import annotations

from src.quality.base import ExpectationResolver


class GeExpectationResolver(ExpectationResolver):
    def __init__(self, gxe) -> None:
        self._gxe = gxe

    def resolve(self, type_name: str) -> type:

        pascal_case = "".join(x.capitalize() for x in type_name.split("_"))

        if hasattr(self._gxe, pascal_case):
            return getattr(self._gxe, pascal_case)

        if hasattr(self._gxe, type_name):
            return getattr(self._gxe, type_name)

        raise AttributeError(f"Regra de qualidade '{type_name}' não suportada pelo motor GE.")
