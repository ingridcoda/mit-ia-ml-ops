"""
modeling/tracker.py — Rastreamento de experimentos e Governança no MLflow.
"""
import json

import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    def log_experimento(self, model_name, metrics, params, artifacts):
        """Registra métricas técnicas e parâmetros no MLflow."""

        tecnica = params.get('selected_reducer') or params.get('strategy') or params.get('reducer') or "PASSTHROUGH"
        nome_amigavel = f"Pipeline_{tecnica.upper()}"

        mlflow.set_tag("mlflow.runName", nome_amigavel)
        mlflow.set_tag("model_type", model_name)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.log_params(params)

        for art_path in artifacts.values():
            if art_path:
                mlflow.log_artifact(str(art_path))

    def salvar_modelo(self, model, model_name="CreditRiskModel", artifact_path="model"):
        """
        Salva o modelo no MLflow, registra e atualiza o alias de produção.
        Resolve segurança do skops e avisos de depreciação.
        """
        trusted_types = [
            "numpy.dtype",
            "src.modeling.reducer.FeatureReducer",
            "src.preprocessing.transformers.stateful.StandardScalerTransformer",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier"
        ]

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            registered_model_name=model_name,
            serialization_format="skops",
            skops_trusted_types=trusted_types
        )

        versao_criada = model_info.registered_model_version

        if versao_criada:
            client = MlflowClient(tracking_uri="sqlite:///mlruns.db")
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=str(versao_criada)
            )
            print(f"🚀 [CD Automático] Alias 'production' assinado para a versão {versao_criada} do '{model_name}'.")

        return model_info

    def salvar_resumo_json(self, best_model, holdout_metrics, path):
        resumo = {
            "model": best_model,
            "f1_holdout": holdout_metrics['f1'],
            "status": "Finalizado",
            "threshold": holdout_metrics.get('best_threshold')
        }
        with open(path, "w") as f:
            json.dump(resumo, f, indent=4)
