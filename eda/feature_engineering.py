"""feature_engineering.py — New Credit Variables."""
import numpy as np
import pandas as pd


def run(df, config, dirs, logger):
    logger.info("=== Feature Engineering for Credit ===")

    df['Monthly_Installment'] = df['Credit amount'] / df['Duration']

    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 25, 45, 100], labels=['Young', 'Adult', 'Senior'])

    df['log_Credit_amount'] = np.log1p(df['Credit amount'])

    df.head(100).to_csv(dirs['tables'] / "09_enriched_credit_sample.csv")

    logger.info("Features created: Monthly_Installment, Age_Category, log_Credit_amount")
    return df
