"""pivot_tables.py — Tabelas dinâmicas para análise de grupos de risco."""
import pandas as pd


def run(df, config, dirs, logger):
    logger.info("=== Tabelas Dinâmicas — Cruzamento de Variáveis ===")
    target = config['feature_selection']['target']

    # 1. Risco por Emprego (Job) e Moradia (Housing)
    # Mostra a contagem de bons/maus pagadores em cada cruzamento
    pivot_job_housing = pd.crosstab(
        [df['Job'], df['Housing']],
        df[target],
        margins=True
    )
    pivot_job_housing.to_csv(dirs['tables'] / "01_pivot_job_housing_risk.csv")

    # 2. Estatísticas de Valor de Crédito por Propósito (Purpose)
    pivot_purpose = df.groupby('Purpose')['Credit amount'].agg(['mean', 'median', 'std', 'count'])
    pivot_purpose.to_csv(dirs['tables'] / "02_stats_by_purpose.csv")

    # 3. Taxa de "Bad Risk" por Categoria de Conta (Checking account)
    # Criamos uma dummy temporária para calcular a média (taxa)
    df_temp = df.copy()
    df_temp['is_bad'] = (df_temp[target] == 'bad').astype(int)
    bad_rate_checking = df_temp.groupby('Checking account')['is_bad'].mean().sort_values(ascending=False)
    bad_rate_checking.to_csv(dirs['tables'] / "03_bad_rate_by_checking_account.csv")

    return {"status": "success"}
