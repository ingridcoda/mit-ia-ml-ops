# %%
# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2: Qualidade de Dados com Great Expectations
# ─────────────────────────────────────────────────────────────────────────────
#
# Responsabilidades:
#   1. Validar o schema do arquivo credit_risk.parquet.
#   2. Verificar integridade (expect_column_values_to_be_in_set para 'Risk').
#   3. Gerar relatório técnico em outputs/quality/.
#
# Configurações: config/quality.yaml
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext

# O contexto carrega o quality.yaml e despacha a validação GE
context = PipelineContext.from_notebook(__file__)
context.run_step("quality")
