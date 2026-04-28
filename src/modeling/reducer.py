"""
modeling/reducer.py — Implementação de Redução de Dimensionalidade (PCA e LDA).
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='passthrough', n_components=None):
        self.strategy = strategy
        self.n_components = n_components

    def fit(self, X, y=None):
        if self.strategy == 'pca':

            n = self.n_components if self.n_components else 0.95
            self.reducer_ = PCA(n_components=n)
            self.reducer_.fit(X)
        elif self.strategy == 'lda':

            self.reducer_ = LDA(n_components=1)
            self.reducer_.fit(X, y)
        return self

    def transform(self, X):
        if self.strategy in ['pca', 'lda'] and hasattr(self, 'reducer_'):
            return self.reducer_.transform(X)
        return X
