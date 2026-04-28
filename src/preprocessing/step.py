"""
preprocessing/step.py — Implementação da Etapa de Pré-processamento.
"""
import pandas as pd

from src.core.base import PipelineStep
from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder


class PreprocessingStep(PipelineStep):
    """
    Executa a limpeza e engenharia de features para o domínio de Crédito.
    """

    def __init__(self, context):
        super().__init__(context)

        self.context = context
        self.config = context.preprocessing_cfg
        self.logger = context.logger

    def run(self):
        self.logger.info("Lendo dataset processado para Engenharia de Features...")

        df = pd.read_parquet(self.context.output_path)

        builder = PreprocessingPipelineBuilder(self.config, self.logger)
        pipeline = builder.build()

        self.logger.info("Iniciando transformações de domínio (Crédito)...")
        df_transformed = pipeline.fit_transform(df)

        output_path = self.context.features_dir / "credit_features.parquet"

        df_transformed.to_parquet(output_path, index=False)

        self.logger.info(f"Features salvas em: {output_path} | Shape: {df_transformed.shape}")
