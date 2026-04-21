"""
preprocessing/pipeline_builder.py — Orquestrador Completo do Pipeline de Features.
"""
from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import (
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    CategoricalEncoder,
    FeatureInteractionTransformer,
    CreditFeatureSelector
)


class PreprocessingPipelineBuilder:
    """
    Monta o sklearn.Pipeline completo para o domínio de Risco de Crédito.
    """

    def __init__(self, config: dict, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger

    def build(self) -> Pipeline:
        """
        Instancia os transformadores seguindo a ordem de dependência lógica:
        1. Flags e Encoders -> 2. Razões e Interações -> 3. Logs -> 4. Seleção.
        """
        etapas = [
            # 1. Tratamento de Variáveis Categóricas (One-Hot / Ordinal)
            ("encoding_categorico", CategoricalEncoder(
                enc_config=self.config.get("categorical_encoding", []),
                logger=self.logger
            )),

            # 2. Criação de Flags Binárias (Thresholds de Risco)
            ("flags_binarias", BinaryFlagTransformer(
                flags=self.config.get("binary_flags", []),
                logger=self.logger
            )),

            # 3. Engenharia de Razões (Ex: Credit Amount / Duration)
            ("razoes_financeiras", RatioFeatureTransformer(
                ratios=self.config.get("ratio_features", []),
                logger=self.logger
            )),

            # 4. Interações Complexas (Ex: Age * Credit Amount)
            ("interacoes_features", FeatureInteractionTransformer(
                interaction_config=self.config.get("feature_interactions", []),
                logger=self.logger
            )),

            # 5. Estabilização de Variância (Log1p)
            ("transformacao_log", LogTransformer(
                columns=self.config.get("log_transform", {}).get("columns", []),
                logger=self.logger
            )),

            # 6. Seleção do Vetor Final de Features (Drop de colunas brutas)
            ("selecao_final", CreditFeatureSelector(
                features_to_keep=self.config.get("feature_selection", {}).get("features_to_keep", []),
                logger=self.logger
            ))
        ]

        if self.logger:
            self.logger.info("PipelineBuilder: Orquestrador configurado com %d transformadores.", len(etapas))

        return Pipeline(etapas)
