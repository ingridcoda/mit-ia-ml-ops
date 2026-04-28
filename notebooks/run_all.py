"""executa_tudo.py — Complete orchestrator for the credit risk pipeline."""

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep
from src.modeling.step import ModelingStep
from src.utils.logger import get_logger


def get_next_version(base_path="outputs/modeling"):
    """Identifies the next version based on existing folders."""
    path = Path(base_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return "v1"
    versions = [int(re.search(r'v(\d+)', p.name).group(1)) for p in path.iterdir() if re.search(r'v(\d+)', p.name)]
    return f"v{max(versions) + 1}" if versions else "v1"


def run_full_pipeline():
    parser = argparse.ArgumentParser(description="Credit Risk MLOps Pipeline")
    parser.add_argument(
        "--reducer",
        type=str,
        default="passthrough",
        choices=["pca", "lda", "passthrough"],
        help="Reduction technique: pca, lda or passthrough."
    )
    args = parser.parse_args()

    logger = get_logger("FullPipeline", {"level": "INFO"})
    VERSION = get_next_version()

    logger.info(f"🚀 Starting Pipeline {VERSION} | Reducer: {args.reducer.upper()}")

    try:
        context = PipelineContext.from_notebook(__file__)

        context.params = {
            'current_version': VERSION,
            'selected_reducer': args.reducer
        }

        logger.info(f"--- Step 1 & 2: Ingestion and Quality ---")
        context.run_step("ingestion")
        context.run_step("quality")

        logger.info(f"--- Step 3: Preprocessing ---")
        PreprocessingStep(context).run()

        logger.info(f"--- Step 4: Modeling and Tracking ---")
        ModelingStep(context).run()

        logger.info(f"✅ Pipeline {VERSION} ({args.reducer}) completed successfully!")

    except Exception as e:
        logger.error(f"❌ Failure in {VERSION}: {str(e)}")
        raise


if __name__ == "__main__":
    run_full_pipeline()
