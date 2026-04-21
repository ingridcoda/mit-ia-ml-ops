"""
utils/pipeline_utils.py — Cadeia de pré-processamento para inferência de crédito.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Configurações de caminhos
_UTILS_DIR = Path(__file__).resolve().parent
_APP_DIR = _UTILS_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
_PARQUET_FEATURES = _PROJECT_ROOT / "data" / "features" / "credit_features.parquet"

# Lista exata de colunas que o seu modelo XGBoost espera (a ordem importa!)
_FEATURES_TO_KEEP = [
    "Age", "Job", "Credit amount", "log_Credit amount",
    "Duration", "checking_acc_encoded", "is_high_risk_amount"
]


def preprocessar_entradas(raw_data: dict) -> pd.DataFrame:
    """Transforma o dicionário do formulário em um DataFrame pronto para o modelo."""
    df = pd.DataFrame([raw_data])

    # 1. Transformação Logarítmica (Valor do Crédito)
    df["log_Credit amount"] = np.log1p(df["Credit amount"])

    # 2. Flag Binária: Valor de Crédito muito alto
    df["is_high_risk_amount"] = (df["Credit amount"] > 5000).astype(int)

    # 3. Encoding Ordinal para Checking Account
    # Mapeamento conforme os dados originais do German Credit
    mapa_checking = {"little": 0, "moderate": 1, "rich": 2, "NA": 3}
    df["checking_acc_encoded"] = df["Checking account"].map(mapa_checking).fillna(3)

    # 4. Seleção e Reindexação: Garante que o DF final tenha apenas as colunas do treino
    df_final = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    return df_final


def obter_parquet_features():
    """Retorna o caminho do arquivo parquet para monitoramento."""
    return _PARQUET_FEATURES
