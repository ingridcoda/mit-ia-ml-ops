# %%
# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1: Ingestão de Dados — Risco de Crédito (UCI)
# ─────────────────────────────────────────────────────────────────────────────
#
# Responsabilidades:
#   1. Download do dataset 'german-credit' via API do Kaggle/UCI.
#   2. Conversão dos arquivos brutos para Parquet (data/processed/credit_risk.parquet).
#   3. Garantia de persistência segura e eficiente.
#
# Configurações: config/data.yaml e config/pipeline.yaml
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no path para importar src/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext

# Inicializa o contexto e executa a ingestão definida no data.yaml
context = PipelineContext.from_notebook(__file__)
context.run_step("ingestion")
