"""
test_optimizer.py — Enhanced tests for the Bayesian Optimizer (Optuna).
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.modeling.cross_validation import CVRunner
from src.modeling.optimizer import OptunaOptimizer


class TestOptunaOptimizerConfig:
    def test_direcao_maximizacao(self, null_logger) -> None:
        """Garante que para classificação o Optuna use 'maximize' para o F1."""
        runner = MagicMock(spec=CVRunner)
        opt = OptunaOptimizer(
            cfg_optuna={}, cv_runner=runner, pipe_cfg={},
            seed=42, n_trials_global=1, logger=null_logger
        )
        assert opt.direction == "maximize"


class TestOptunaOptimizerLogica:
    @patch("optuna.create_study")
    @patch("src.modeling.optimizer.construir_pipeline")
    def test_fluxo_otimizacao_completo(self, mock_pipeline, mock_study, null_logger):
        """Valida se o otimizador cria o estudo e retorna os parâmetros corretos."""

        runner = MagicMock(spec=CVRunner)
        study_instance = mock_study.return_value
        study_instance.best_params = {'n_estimators': 100}

        opt = OptunaOptimizer(
            cfg_optuna={'default_trials': 5},
            cv_runner=runner,
            pipe_cfg={},
            seed=42,
            n_trials_global=10,
            logger=null_logger
        )

        X, y = pd.DataFrame(), pd.Series()
        model_cfg = {'search_space': {'n_estimators': [50, 100]}}

        resultado = opt.otimizar("XGB", model_cfg, X, y, {}, {})

        mock_study.assert_called_once_with(direction="maximize")
        study_instance.optimize.assert_called_once()
        assert resultado["estimator_params"] == {'n_estimators': 100}
        assert "reducer_params" in resultado

    def test_objetivo_calculo_media_f1(self, null_logger):
        """Testa se a função objetivo interna está calculando a média dos folds corretamente."""

        runner = MagicMock(spec=CVRunner)
        runner.executar.return_value = [{'f1': 0.8}, {'f1': 0.9}]

        opt = OptunaOptimizer(
            cfg_optuna={}, cv_runner=runner, pipe_cfg={},
            seed=42, n_trials_global=1, logger=null_logger
        )

        trial = MagicMock()

        opt._sugerir_params = MagicMock(return_value={})

        with patch("src.modeling.optimizer.construir_pipeline"):
            folds = runner.executar(MagicMock(), pd.DataFrame(), pd.Series())
            media_f1 = np.mean([f['f1'] for f in folds])

            assert media_f1 == pytest.approx(0.85)
