# %%
# ─────────────────────────────────────────────────────────────────────────────
# Etapa 3: Pré-processamento e Engenharia de Features de Crédito
# ─────────────────────────────────────────────────────────────────────────────
#
# Entrada : data/processed/credit_risk.parquet
# Saída   : data/features/credit_features.parquet
#
# Transformações principais:
#   1. Encoding Ordinal: Status de conta corrente e poupança.
#   2. Flags: Identificação de valores de crédito atípicos.
#   3. Log Transform: Tratamento de assimetria no valor do empréstimo.
#   4. Seleção: Manter apenas preditores financeiros relevantes.
#
# ⚠ ANTI-LEAKAGE: Imputação e Escalonamento ficam para a etapa de modelagem.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

# Inicializa contexto e executa o PreprocessingStep configurado
ctx = PipelineContext.from_notebook(__file__)
step = PreprocessingStep(ctx)
step.run()
