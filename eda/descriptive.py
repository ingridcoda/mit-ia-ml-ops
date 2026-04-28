"""descriptive.py — Descriptive Statistics for Credit Risk."""
import json


def run(df, config, dirs, logger):
    logger.info("=== Descriptive Statistics — Credit Risk ===")
    target = config['feature_selection']['target']

    info = {
        "shape": df.shape,
        "target_distribution": df[target].value_counts(normalize=True).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    with open(dirs['stats'] / "01_basic_info.json", "w") as f:
        json.dump(info, f, indent=4)

    stats = df.describe().T
    stats.to_csv(dirs['stats'] / "02_descriptive_stats.csv")

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
