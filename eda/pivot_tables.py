"""pivot_tables.py — Pivot Tables for Risk Group Analysis."""
import pandas as pd


def run(df, config, dirs, logger):
    logger.info("=== Tabelas Dinâmicas — Cruzamento de Variáveis ===")
    target = config['feature_selection']['target']

    pivot_job_housing = pd.crosstab(
        [df['Job'], df['Housing']],
        df[target],
        margins=True
    )
    pivot_job_housing.to_csv(dirs['tables'] / "01_pivot_job_housing_risk.csv")

    pivot_purpose = df.groupby('Purpose')['Credit amount'].agg(['mean', 'median', 'std', 'count'])
    pivot_purpose.to_csv(dirs['tables'] / "02_stats_by_purpose.csv")

    df_temp = df.copy()
    df_temp['is_bad'] = (df_temp[target] == 'bad').astype(int)
    bad_rate_checking = df_temp.groupby('Checking account')['is_bad'].mean().sort_values(ascending=False)
    bad_rate_checking.to_csv(dirs['tables'] / "03_bad_rate_by_checking_account.csv")

    return {"status": "success"}
