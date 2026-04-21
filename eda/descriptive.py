"""descriptive.py — Estatísticas descritivas para Risco de Crédito."""
import json


def run(df, config, dirs, logger):
    logger.info("=== Estatísticas Descritivas — Risco de Crédito ===")
    target = config['feature_selection']['target']

    # 1. Info básica e counts do Target
    info = {
        "shape": df.shape,
        "target_distribution": df[target].value_counts(normalize=True).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    with open(dirs['stats'] / "01_basic_info.json", "w") as f:
        json.dump(info, f, indent=4)

    # 2. Estatísticas numéricas (Idade, Valor do Crédito, Duração)
    stats = df.describe().T
    stats.to_csv(dirs['stats'] / "02_descriptive_stats.csv")

    # 3. Análise de Outliers no Valor do Crédito (IQR)
    q1 = df['Credit amount'].quantile(0.25)
    q3 = df['Credit amount'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['Credit amount'] < (q1 - 1.5 * iqr)) | (df['Credit amount'] > (q3 + 1.5 * iqr))]

    outlier_info = {
        "total_outliers_credit": len(outliers),
        "percentage": len(outliers) / len(df)
    }
    with open(dirs['stats'] / "05_outliers_iqr.json", "w") as f:
        json.dump(outlier_info, f, indent=4)

    return {"status": "success"}
