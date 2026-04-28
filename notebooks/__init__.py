"""
notebooks/ — Orchestration Scripts for the Credit Risk Pipeline.

This module contains the entry points for executing the pipeline steps.
The files use cell syntax (# %%) for compatibility with
VS Code Interactive Window and Jupyter.

Recommended execution order:
    1. ingestion.py         — Loading raw data (Kaggle/UCI).
    2. quality.py        — Validation via Great Expectations.
    3. preprocessing.py — Feature engineering (Stateless).
    4. modeling.py        — Training, HPO (Optuna) and Registration (MLflow).
    5. run_all.py     — Complete orchestrator for one-click execution.
"""

__version__ = "1.0.0"
