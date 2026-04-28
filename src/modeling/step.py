"""
modeling/step.py — Maestro integrado com parâmetros dinâmicos e persistência versionada.
"""
import json

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from src.modeling.evaluator import HoldoutEvaluator
from src.modeling.model_factory import construir_pipeline
from src.modeling.tracker import MLflowTracker


class ModelingStep:
    def __init__(self, context):
        self.context = context
        self.logger = context.logger

    def run(self):
        params = getattr(self.context, 'params', {})
        version = params.get('current_version', 'v1')
        reducer_strategy = params.get('selected_reducer', 'passthrough')

        feature_path = self.context.features_dir / "credit_features.parquet"
        df = pd.read_parquet(feature_path)
        target = self.context.modeling_cfg['target_column']
        X = df.drop(columns=[target])
        y = df[target].map({'good': 0, 'bad': 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scale_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        best_model_cfg = self.context.modeling_cfg['best_model'].copy()
        dict_params_modelo = {'scale_pos_weight': scale_weight, 'random_state': 42}
        dict_params_reducer = {'strategy': reducer_strategy}

        pipeline = construir_pipeline(
            model_cfg=best_model_cfg,
            params_modelo=dict_params_modelo,
            params_reducer=dict_params_reducer,
            pipe_cfg=self.context.modeling_cfg.get('pipeline_params', {})
        )

        run_name = f"Experimento_{version}_{reducer_strategy.upper()}"
        with mlflow.start_run(run_name=run_name):
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
            cv_f1_mean = cv_scores.mean()

            pipeline.fit(X_train, y_train)

            evaluator = HoldoutEvaluator(self.logger)
            metrics = evaluator.avaliar(pipeline, X_test, y_test)
            robustez = evaluator.diagnosticar_robustez(cv_f1_mean, metrics['f1'])

            tracker = MLflowTracker()
            tracker.log_experimento(
                model_name=f"XGB_{reducer_strategy.upper()}",
                metrics={**metrics, 'cv_f1_mean': cv_f1_mean},
                params={**dict_params_modelo, 'version': version, 'robustez': robustez,
                        'selected_reducer': reducer_strategy},
                artifacts={}
            )

            model_dir = self.context.root_dir / "outputs" / "modeling" / version
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_dir / f"model_{version}.joblib")

            json_metrics = {**metrics, "selected_reducer": reducer_strategy}
            with open(model_dir / f"model_{version}_metadata.json", "w") as f:
                json.dump(json_metrics, f, indent=4)

            self.logger.info(f"Integrando com Model Registry (CD Automático)...")
            tracker.salvar_modelo(
                model=pipeline,
                model_name="CreditRiskModel",
                artifact_path=f"model_{reducer_strategy.lower()}"
            )

        self.logger.info(f"ModelingStep finalizado: {version} ({reducer_strategy})")
        return pipeline
