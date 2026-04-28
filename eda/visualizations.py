"""visualizations.py — Visualizations for Risk Classification."""
import matplotlib.pyplot as plt
import seaborn as sns


def run(df, config, dirs, logger):
    logger.info("=== Gerando Gráficos de Risco ===")
    target = config['feature_selection']['target']

    plt.figure(figsize=(8, 5))
    sns.countplot(x=target, data=df, palette='viridis')
    plt.title("Distribuição de Bons (good) e Maus (bad) Pagadores")
    plt.savefig(dirs['figures'] / "fig_01_target_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target, y='Credit amount', data=df)
    plt.title("Distribuição do Valor do Crédito por Categoria de Risco")
    plt.savefig(dirs['figures'] / "fig_02_credit_value_by_risk.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="Age", hue=target, fill=True)
    plt.title("Densidade de Idade por Risco de Crédito")
    plt.savefig(dirs['figures'] / "fig_03_age_density.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='RdBu', fmt=".2f")
    plt.title("Correlação entre Variáveis Numéricas")
    plt.savefig(dirs['figures'] / "fig_04_correlation_matrix.png")
    plt.close()

    return {"status": "success"}
