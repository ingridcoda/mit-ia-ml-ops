# %% [markdown]
# # Orquestrador Completo do Pipeline de Crédito
# Este script executa todas as etapas em sequência, garantindo que o modelo
# final seja registrado no MLflow com um único clique.

# %%
import sys
from pathlib import Path

# Configuração de Path para garantir acesso à src/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep
from src.modeling import ModelingStep
from src.utils.logger import get_logger


# %%
def run_full_pipeline():
    logger = get_logger("FullPipeline", {"level": "INFO"})
    logger.info("🚀 Iniciando execução completa do pipeline de Risco de Crédito...")

    try:
        # 1. Contexto e Ingestão
        context = PipelineContext.from_notebook(__file__)

        logger.info("--- Etapa 1: Ingestão ---")
        context.run_step("ingestion")

        # 2. Qualidade
        logger.info("--- Etapa 2: Qualidade ---")
        context.run_step("quality")

        # 3. Pré-processamento
        logger.info("--- Etapa 3: Pré-processamento ---")
        PreprocessingStep(context).run()

        # 4. Modelagem (HPO + MLflow)
        logger.info("--- Etapa 4: Modelagem ---")
        ModelingStep(context).run()

        logger.info("✅ Pipeline finalizado com sucesso! Modelo disponível no Registry.")

    except Exception as e:
        logger.error(f"❌ Falha crítica no pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    run_full_pipeline()
