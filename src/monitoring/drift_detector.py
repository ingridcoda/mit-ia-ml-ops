"""
src/monitoring/drift_detector.py — Detecção de Drift via Teste Kolmogorov-Smirnov.
"""
import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:
    def __init__(self, df_treino: pd.DataFrame):
        self.df_treino = df_treino

    def detectar_drift(self, df_producao: pd.DataFrame, p_value_threshold=0.05):
        """Compara distribuições de features numéricas."""
        resultados = {}
        features_num = self.df_treino.select_dtypes(include=['number']).columns

        for col in features_num:
            if col in df_producao.columns:
                stat, p_val = ks_2samp(self.df_treino[col], df_producao[col], method='asymp')
                resultados[col] = {
                    "p_value": float(p_val),
                    "drift_detectado": bool(p_val < p_value_threshold)
                }
        return resultados
