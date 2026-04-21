"""__init__.py — Exporta a API pública do pacote EDA."""
from eda import (
    descriptive,
    visualizations,
    pivot_tables,
    statistical_tests,
    feature_engineering,
    clustering
)

__all__ = [
    "descriptive",
    "visualizations",
    "pivot_tables",
    "statistical_tests",
    "feature_engineering",
    "clustering"
]
