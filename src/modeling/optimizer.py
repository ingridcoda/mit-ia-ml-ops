"""
modeling/optimizer.py — Otimização Bayesiana focada em F1-Score.
"""
from __future__ import annotations

import numpy as np
import optuna

from src.modeling.base import BaseOptimizer
from src.modeling.model_factory import construir_pipeline


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, cfg_optuna, cv_runner, pipe_cfg, seed, n_trials_global, logger):
        self.cv_runner = cv_runner
        self.direction = "maximize"  # Vital para classificação (F1)
        self.n_trials_global = n_trials_global

    def otimizar(self, model_name, model_cfg, X, y, pipe_cfg, feat_red_cfg):
        def objective(trial):
            params = self._sugerir_params(trial, model_cfg.get('search_space', {}))
            pipeline = construir_pipeline(model_cfg, params, None, pipe_cfg)

            folds = self.cv_runner.executar(pipeline, X, y)
            return np.mean([f['f1'] for f in folds])

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials_global)
        return {"estimator_params": study.best_params, "reducer_params": {}}
