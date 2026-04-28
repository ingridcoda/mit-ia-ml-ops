"""__init__.py — Exports the public API of the EDA package."""
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
