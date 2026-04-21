"""
modeling/step.py — Maestro integrado com CV e MLflow.
"""
import json

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from src.core.base import PipelineStep
from src.modeling.evaluator import HoldoutEvaluator
from src.modeling.model_factory import construir_pipeline
from src.modeling.tracker import MLflowTracker


class ModelingStep(PipelineStep):
    def __init__(self, context):
        super().__init__(context)
        self.context = context
        self.logger = context.logger

    def run(self):
        # 1. Preparação de Dados
        feature_path = self.context.features_dir / "credit_features.parquet"
        df = pd.read_parquet(feature_path)
        target = self.context.modeling_cfg['target_column']
        X = df.drop(columns=[target])
        y = df[target].map({'good': 0, 'bad': 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # O XGBoost lida bem com NaNs, mas o LDA não.
        # Preenchemos os nulos com a mediana aprendida estritamente no treino
        # para evitar vazamento de dados.
        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        # 2. Configuração Dinâmica (Versão LDA + scale_pos_weight)
        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params_modelo = {'scale_pos_weight': scale_weight}
        params_reducer = {'strategy': 'lda'}  # Justificado na Parte 4 como técnica supervisionada

        # # 2. Configuração Dinâmica (Versão sem redução para maximizar performance)
        # scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        # params_modelo = {'scale_pos_weight': scale_weight}
        # params_reducer = {'strategy': 'passthrough'} # Alterado para 'passthrough' após provarmos que LDA/PCA pioram o F1

        pipeline = construir_pipeline(
            self.context.modeling_cfg['best_model'],
            params_modelo,
            params_reducer,
            self.context.modeling_cfg['pipeline_params']
        )

        # 3. Treinamento e Rastreamento (Parte 3 da Rubrica)
        with mlflow.start_run(run_name="Credit_Final_Model"):
            # Validação Cruzada (Exigência da Rubrica)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
            cv_f1_mean = cv_scores.mean()

            pipeline.fit(X_train, y_train)

            # 4. Avaliação e Diagnóstico de Robustez
            evaluator = HoldoutEvaluator(self.logger)
            metrics = evaluator.avaliar(pipeline, X_test, y_test)
            robustez = evaluator.diagnosticar_robustez(cv_f1_mean, metrics['f1'])

            # 5. Registro no MLflow
            tracker = MLflowTracker()
            tracker.log_experimento(
                model_name="XGB_LDA",
                metrics={**metrics, 'cv_f1_mean': cv_f1_mean},
                params={**params_modelo, **params_reducer, 'robustez': robustez},
                artifacts={}
            )

            # 6. Salvamento de Artefatos
            model_dir = self.context.root_dir / "outputs" / "modeling"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_dir / "credit_model_v1.joblib")
            with open(model_dir / "model_metadata.json", "w") as f:
                json.dump(metrics, f, indent=4)

        self.logger.info(f"🚀 Treino concluído. Robustez: {robustez} | F1: {metrics['f1']:.4f}")
