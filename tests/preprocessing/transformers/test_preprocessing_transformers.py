"""
test_preprocessing_transformers.py — Validação de TODOS os transformadores de crédito.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.transformers import (
    FeatureInteractionTransformer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    CategoricalEncoder,
    CreditFeatureSelector
)
from src.preprocessing.transformers.stateful import (
    ZScoreScaler,
    GroupWiseImputer,
    StandardScalerTransformer
)


def test_z_score_scaler_matematica(null_logger):
    """Garante que o ZScoreScaler centraliza na média 0 e std 1."""
    df = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})
    scaler = ZScoreScaler(columns=["feat"], logger=null_logger)
    df_out = scaler.fit_transform(df)

    assert df_out["feat"].mean() == pytest.approx(0.0, abs=1e-9)
    assert df_out["feat"].std() == pytest.approx(1.0, abs=1e-9)


def test_group_wise_imputer(null_logger):
    """Testa se preenche nulos baseado no grupo."""
    df = pd.DataFrame({
        "Job": [1, 1, 2, 2],
        "Amount": [100, np.nan, 500, 500]
    })
    imputer = GroupWiseImputer(group_col="Job", target_col="Amount", logger=null_logger)
    df_out = imputer.fit_transform(df)
    assert df_out.loc[1, "Amount"] == 100


def test_standard_scaler_transformer_integracao(null_logger):
    """Garante que o wrapper do StandardScaler mantém a estrutura do DataFrame."""
    df = pd.DataFrame({"Idade": [20, 30, 40], "Renda": [1000, 2000, 3000]})
    scaler = StandardScalerTransformer(columns=["Idade", "Renda"])
    df_out = scaler.fit_transform(df)

    assert isinstance(df_out, pd.DataFrame)
    assert df_out["Idade"].mean() == pytest.approx(0.0, abs=1e-7)


def test_feature_interaction_transformer(null_logger):
    """Valida a criação de interações multiplicativas."""
    df = pd.DataFrame({"Age": [2], "Amount": [50]})
    config = [{"name": "age_amount_risk", "columns": ["Age", "Amount"]}]
    transformer = FeatureInteractionTransformer(config, logger=null_logger)
    df_out = transformer.transform(df)
    assert df_out["age_amount_risk"].iloc[0] == 100


def test_binary_flag_transformer(null_logger):
    """Testa se a flag binária de risco financeiro é criada corretamente."""
    df = pd.DataFrame({"Credit amount": [5000, 10000, 15000]})
    config = [{"column": "Credit amount", "value": 10000, "new_column": "is_high_risk_amount"}]
    transformer = BinaryFlagTransformer(config, logger=null_logger)
    df_out = transformer.transform(df)
    assert df_out["is_high_risk_amount"].iloc[0] == 0
    assert df_out["is_high_risk_amount"].iloc[2] == 1


def test_ratio_feature_transformer(null_logger):
    """Valida o cálculo de proporções (ex: Parcela = Valor / Duração)."""
    df = pd.DataFrame({"Credit amount": [1000, 2400], "Duration": [10, 24]})
    config = [{"name": "installment", "numerator": "Credit amount", "denominator": "Duration"}]
    transformer = RatioFeatureTransformer(config, logger=null_logger)
    df_out = transformer.transform(df)
    assert df_out["installment"].iloc[0] == 100.0
    assert df_out["installment"].iloc[1] == 100.0


def test_log_transformer(null_logger):
    """Garante que a transformação logarítmica (log1p) está correta para assimetria."""
    df = pd.DataFrame({"Amount": [0, 99]})
    transformer = LogTransformer(columns=["Amount"], logger=null_logger)
    df_out = transformer.transform(df)
    assert "log_Amount" in df_out.columns
    assert df_out["log_Amount"].iloc[0] == 0.0


def test_categorical_encoder(null_logger):
    """Testa o mapeamento ordinal de strings para números."""
    df = pd.DataFrame({"Checking": ["little", "rich", "little"]})
    config = [{
        "column": "Checking",
        "ordinal_column": "Checking_enc",
        "ordinal_map": {"little": 0, "rich": 2}
    }]
    transformer = CategoricalEncoder(config, logger=null_logger)
    df_out = transformer.transform(df)
    assert df_out["Checking_enc"].iloc[0] == 0
    assert df_out["Checking_enc"].iloc[1] == 2


def test_credit_feature_selector(null_logger):
    """Valida se o filtro final descarta o lixo e mantém as colunas vitais."""
    df = pd.DataFrame({"Age": [30], "Lixo": [999], "Risk": [0]})
    features = ["Age", "Risk"]
    transformer = CreditFeatureSelector(features_to_keep=features, logger=null_logger)
    df_out = transformer.transform(df)
    assert list(df_out.columns) == ["Age", "Risk"]
    assert "Lixo" not in df_out.columns
