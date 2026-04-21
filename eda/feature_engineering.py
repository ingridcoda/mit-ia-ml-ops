"""feature_engineering.py — Novas Variáveis de Crédito."""
import numpy as np
import pandas as pd


def run(df, config, dirs, logger):
    logger.info("=== Engenharia de Features para Crédito ===")

    # 1. Parcela Mensal Estimada (Credit Amount / Duration)
    df['Monthly_Installment'] = df['Credit amount'] / df['Duration']

    # 2. Idade Binada (Jovem, Adulto, Sênior)
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 25, 45, 100], labels=['Young', 'Adult', 'Senior'])

    # 3. Log do Valor do Crédito (para reduzir skewness)
    df['log_Credit_amount'] = np.log1p(df['Credit amount'])

    # Salva amostra enriquecida
    df.head(100).to_csv(dirs['tables'] / "09_enriched_credit_sample.csv")

    logger.info("Features criadas: Monthly_Installment, Age_Category, log_Credit_amount")
    return df
