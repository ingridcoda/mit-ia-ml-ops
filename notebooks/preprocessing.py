"""preprocessamento.py — Preprocessing step for feature engineering."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

ctx = PipelineContext.from_notebook(__file__)
step = PreprocessingStep(ctx)
step.run()
