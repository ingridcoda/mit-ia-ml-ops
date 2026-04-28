"""
modeling/model_factory.py — Fábrica de pipelines profissional e resiliente.
"""
import importlib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling.reducer import FeatureReducer
from src.preprocessing.transformers.stateful import StandardScalerTransformer


def construir_pipeline(model_cfg, params_modelo, params_reducer, pipe_cfg):
    """
    Constrói um pipeline end-to-end garantindo a integridade dos tipos de dados.
    """
    steps = []

    imputer = SimpleImputer(strategy='median')
    imputer.set_output(transform="pandas")
    steps.append(('imputer', imputer))

    cols_scale = pipe_cfg.get('scaling', {}).get('columns', [])
    if cols_scale:
        steps.append(('scaler', StandardScalerTransformer(columns=cols_scale)))

    steps.append(('reducer', FeatureReducer(**(params_reducer or {}))))

    modulo = importlib.import_module(model_cfg['module'])
    cls = getattr(modulo, model_cfg['class'])
    final_params = {**model_cfg.get('default_params', {}), **(params_modelo or {})}
    steps.append(('estimator', cls(**final_params)))

    return Pipeline(steps)
