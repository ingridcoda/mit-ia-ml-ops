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
        # Salvando o contexto explicitamente para uso nos métodos da classe
        self.context = context
        self.config = context.preprocessing_cfg
        self.logger = context.logger

    def run(self):
        self.logger.info("Lendo dataset processado para Engenharia de Features...")

        # 1. Carregamento do Parquet gerado na Ingestão
        df = pd.read_parquet(self.context.output_path)

        # 2. Construção do Pipeline via Builder
        builder = PreprocessingPipelineBuilder(self.config, self.logger)
        pipeline = builder.build()

        # 3. Execução das Transformações (Fit & Transform)
        self.logger.info("Iniciando transformações de domínio (Crédito)...")
        df_transformed = pipeline.fit_transform(df)

        # 4. Salvamento do Vetor de Features
        # Definindo saída em data/features/ conforme a estrutura do projeto
        output_path = self.context.features_dir / "credit_features.parquet"

        df_transformed.to_parquet(output_path, index=False)

        self.logger.info(f"Features salvas em: {output_path} | Shape: {df_transformed.shape}")
