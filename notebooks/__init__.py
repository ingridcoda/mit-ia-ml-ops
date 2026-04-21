"""
notebooks/ — Scripts de Orquestração do Pipeline de Risco de Crédito.

Este módulo contém os pontos de entrada para execução das etapas do pipeline.
Os arquivos utilizam a sintaxe de células (# %%) para compatibilidade com
o VS Code Interactive Window e Jupyter.

Ordem recomendada de execução:
    1. ingestao.py         — Carga dos dados brutos (Kaggle/UCI).
    2. qualidade.py        — Validação via Great Expectations.
    3. preprocessamento.py — Engenharia de features (Stateless).
    4. modelagem.py        — Treino, HPO (Optuna) e Registro (MLflow).
    5. executa_tudo.py     — Orquestrador completo para execução com um clique.
"""

__version__ = "1.0.0"
