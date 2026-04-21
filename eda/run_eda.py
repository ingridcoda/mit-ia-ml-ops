"""run_eda.py — Orquestrador do Pipeline de EDA."""
import time
from pathlib import Path

# Importe os módulos que refatoramos
from eda import descriptive, visualizations, statistical_tests, feature_engineering, clustering, pivot_tables
from src.utils.config_loader import load_yaml
from src.utils.logger import get_logger


def run_eda(base_dir="."):
    pipeline_start = time.time()
    root = Path(base_dir)

    # Carrega configs
    config = load_yaml(root / "config" / "preprocessing.yaml")
    # Nota: Se tiver um eda.yaml específico, use-o aqui

    logger = get_logger("EDA_Pipeline", config.get("logging", {}))

    # Define diretórios de saída
    dirs = {
        "stats": root / "outputs" / "eda" / "stats",
        "figures": root / "outputs" / "eda" / "figures",
        "tables": root / "outputs" / "eda" / "tables"
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # Sequência de Execução
    df = pd.read_parquet(root / "data" / "processed" / "credit_risk.parquet")

    descriptive.run(df, config, dirs, logger)
    visualizations.run(df, config, dirs, logger)
    pivot_tables.run(df, config, dirs, logger)
    statistical_tests.run(df, config, dirs, logger)
    df = feature_engineering.run(df, config, dirs, logger)  # Retorna DF com novas colunas
    clustering.run(df, config, dirs, logger)

    logger.info(f"EDA completo em {time.time() - pipeline_start:.2f}s")


if __name__ == "__main__":
    run_eda()
