"""statistical_tests.py — Independence Tests for Credit."""
import json

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def run(df, config, dirs, logger):
    logger.info("=== Statistical Tests — Chi-Squared and Mann-Whitney ===")
    target = config['feature_selection']['target']

    cat_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account']
    chi2_results = {}

    for feat in cat_features:
        contingency = pd.crosstab(df[feat], df[target])
        chi2, p, dof, ex = chi2_contingency(contingency)
        chi2_results[feat] = {"p-value": p, "significant": p < 0.05}

    with open(dirs['stats'] / "09_chi2_tests.json", "w") as f:
        json.dump(chi2_results, f, indent=4)

    group_good = df[df[target] == 'good']['Credit amount']
    group_bad = df[df[target] == 'bad']['Credit amount']
    stat, p_mw = mannwhitneyu(group_good, group_bad)

    mw_result = {
        "test": "Mann-Whitney U (Credit Amount vs Risk)",
        "p-value": p_mw,
        "significant": p_mw < 0.05
    }
    with open(dirs['stats'] / "10_mann_whitney_test.json", "w") as f:
        json.dump(mw_result, f, indent=4)

    return {"status": "success"}
