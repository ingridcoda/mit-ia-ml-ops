"""model_utils.py — Utilities for loading and managing credit risk models in production."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


def load_credit_model():
    """
    Tenta carregar o modelo de produção do MLflow usando Aliases (@production).
    Utiliza mlflow.sklearn para garantir acesso ao método predict_proba.
    Usa o arquivo .joblib local como fallback de segurança.
    """
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    try:
        model_uri = "models:/CreditRiskModel@production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:

        base_dir = Path(__file__).parent.parent.parent
        model_path = base_dir / "outputs" / "modeling" / "credit_model_v1.joblib"

        if not model_path.exists():
            return None
        return joblib.load(model_path)
