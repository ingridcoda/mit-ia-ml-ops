"""interactions.py — Análise de interações para Risco de Crédito."""
import matplotlib.pyplot as plt
import seaborn as sns


def run(df, config, dirs, logger):
    logger.info("=== Análise de Interações — Taxas de Risco ===")
    target = config['feature_selection']['target']

    # Criamos uma coluna numérica temporária para calcular a taxa (0 para good, 1 para bad)
    df_temp = df.copy()
    df_temp['is_bad'] = (df_temp[target] == 'bad').astype(int)

    # --- 1. Interação 2-Way: Sexo vs Moradia vs Taxa de Calote ---
    plt.figure(figsize=(10, 6))
    interaction_2way = df_temp.pivot_table(
        index='Sex',
        columns='Housing',
        values='is_bad',
        aggfunc='mean'
    )
    sns.heatmap(interaction_2way, annot=True, fmt=".2%", cmap="YlOrRd")
    plt.title("Taxa de Inadimplência: Sexo vs Moradia")
    plt.savefig(dirs['figures'] / "fig_16_interaction_sex_housing.png")
    plt.close()

    # --- 2. Interação 3-Way: Status da Conta vs Conta Poupança (Facetado por Sexo) ---
    # Este gráfico mostra como o perfil financeiro muda entre homens e mulheres
    g = sns.FacetGrid(df_temp, col="Sex", height=5)
    g.map_dataframe(
        lambda data, **kwargs: sns.heatmap(
            data.pivot_table(index='Checking account', columns='Saving accounts', values='is_bad', aggfunc='mean'),
            annot=True, fmt=".0%", cmap="Reds", cbar=False
        )
    )
    g.set_titles("Taxa de Calote - Sexo: {col_name}")
    plt.savefig(dirs['figures'] / "fig_17_interaction_3way_financials.png")
    plt.close()

    # --- 3. Interação Numérica: Idade vs Valor do Crédito (Cores pelo Risco) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Age", y="Credit amount", hue=target, alpha=0.6)
    plt.title("Dispersão: Idade vs Valor do Crédito por Risco")
    plt.savefig(dirs['figures'] / "fig_18_interaction_age_credit.png")
    plt.close()

    # --- Salvar Estatísticas de Interação ---
    interaction_2way.to_csv(dirs['stats'] / "18_interaction_2way_means.csv")

    # 3-way summary (Checking x Saving x Risk)
    df_temp.groupby(['Checking account', 'Saving accounts'])[target].value_counts(normalize=True).unstack().to_csv(
        dirs['stats'] / "19_interaction_3way_stats.csv"
    )

    return {"status": "success"}
