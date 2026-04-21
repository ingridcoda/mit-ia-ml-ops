"""
modeling/model_factory.py — Fábrica de pipelines de classificação.
"""
import importlib

from sklearn.pipeline import Pipeline

from src.modeling.reducer import FeatureReducer
from src.preprocessing.transformers.stateful import StandardScalerTransformer


def construir_pipeline(model_cfg, params_modelo, params_reducer, pipe_cfg):
    steps = []

    # 1. Escalonamento (Stateful - aprende no treino)
    cols_scale = pipe_cfg.get('scaling', {}).get('columns', [])
    if cols_scale:
        # Agora a classe existe e será importada corretamente
        steps.append(('scaler', StandardScalerTransformer(columns=cols_scale)))

    # 2. Redução de Dimensionalidade (PCA ou LDA conforme Rubrica)
    steps.append(('reducer', FeatureReducer(**(params_reducer or {}))))

    # 3. Classificador Final
    modulo = importlib.import_module(model_cfg['module'])
    cls = getattr(modulo, model_cfg['class'])

    # Merge de parâmetros padrão com os parâmetros dinâmicos (ex: scale_pos_weight)
    final_params = {**model_cfg.get('default_params', {}), **(params_modelo or {})}
    steps.append(('estimator', cls(**final_params)))

    return Pipeline(steps)
