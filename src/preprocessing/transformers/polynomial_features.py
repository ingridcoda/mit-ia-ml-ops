"""
transformers/polynomial_features.py — Transformador de Interações e Polinômios.

Cria termos de interação (x1 * x2) ou termos quadráticos (x1^2) para capturar
complexidades no comportamento de crédito que modelos lineares simples perderiam.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class FeatureInteractionTransformer(BaseFeatureTransformer):
    """
    Gera novas features baseadas na interação entre colunas existentes.

    Exemplos em Crédito:
    - Age * Credit Amount : Captura o risco de empréstimos altos para jovens.
    - Credit Amount * Duration : Representa o volume total de exposição financeira.
    """

    def __init__(self, interaction_config: list[dict], logger: Any = None) -> None:
        super().__init__(logger=logger)
        self.interaction_config = interaction_config

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executa as multiplicações configuradas no preprocessing.yaml."""
        X = X.copy()

        for spec in self.interaction_config:
            new_feature_name = spec["name"]
            input_cols = spec["columns"]

            # Validação de existência das colunas
            missing = [c for c in input_cols if c not in X.columns]
            if missing:
                self._warn("Falha ao criar '%s': colunas %s não encontradas.", new_feature_name, missing)
                continue

            # Garantia de que são dados numéricos antes de operar
            if not all(pd.api.types.is_numeric_dtype(X[c]) for c in input_cols):
                self._warn("Falha ao criar '%s': uma ou mais colunas não são numéricas.", new_feature_name)
                continue

            # Lógica de Transformação
            try:
                if len(input_cols) == 1:
                    # Termo quadrático (x^2)
                    X[new_feature_name] = X[input_cols[0]] ** 2
                elif len(input_cols) == 2:
                    # Interação simples (x1 * x2)
                    X[new_feature_name] = X[input_cols[0]] * X[input_cols[1]]
                else:
                    self._warn("Ignorando '%s': suportamos apenas 1 ou 2 colunas por interação.", new_feature_name)
                    continue

                self._log("Feature de interação '%s' gerada com sucesso.", new_feature_name)

            except Exception as e:
                self._warn("Erro inesperado ao processar '%s': %s", new_feature_name, str(e))

        return X
