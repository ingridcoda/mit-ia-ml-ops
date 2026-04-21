"""
modeling/tracker.py — Rastreamento de experimentos no MLflow.
"""
import json

import mlflow


class MLflowTracker:
    def log_experimento(self, model_name, metrics, params, artifacts):
        """Registra métricas técnicas e parâmetros no MLflow."""
        # Registra métricas (F1, Accuracy, Recall, Best Threshold)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Registra parâmetros do modelo e da redução
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)

        for art_path in artifacts.values():
            if art_path:
                mlflow.log_artifact(str(art_path))

    def salvar_resumo_json(self, best_model, holdout_metrics, path):
        resumo = {
            "model": best_model,
            "f1_holdout": holdout_metrics['f1'],
            "status": "Finalizado",
            "threshold": holdout_metrics.get('best_threshold')
        }
        with open(path, "w") as f:
            json.dump(resumo, f, indent=4)
