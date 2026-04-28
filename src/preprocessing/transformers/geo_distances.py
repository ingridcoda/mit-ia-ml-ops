"""transformers/geo_distances.py — No-op para Crédito."""
from src import BaseFeatureTransformer


class GeoDistanceTransformer(BaseFeatureTransformer):
    def transform(self, X, y=None):
        return X
