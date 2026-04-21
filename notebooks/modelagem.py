# %%
# ─────────────────────────────────────────────────────────────────────────────
# Etapa 4: Modelagem e Experimentação MLOps — Classificação de Risco
# ─────────────────────────────────────────────────────────────────────────────
#
# Entrada : data/features/credit_features.parquet
# Saída   : mlruns.db (Tracking SQLite) e outputs/modeling/ (Artefatos)
#
# Diferenciais do Projeto:
#   • Otimização Bayesiana: Optuna buscando maximizar o F1-Score.
#   • Redução de Dimensionalidade: PCA aplicado de forma leak-free.
#   • Ensembles: Combinação de modelos via VotingClassifier.
#   • Robustez: Comparação de métricas entre CV e Holdout.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext
from src.modeling.step import ModelingStep

# Instancia o ModelingStep, que lê o modeling.yaml e orquestra o MLflow
context = PipelineContext.from_notebook(__file__)
step = ModelingStep(context)
step.run()
