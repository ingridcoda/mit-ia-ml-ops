"""modelagem.py — Modeling step for training and optimization."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.context import PipelineContext
from src.modeling.step import ModelingStep

context = PipelineContext.from_notebook(__file__)
context.params = {'current_version': 'v1', 'selected_reducer': 'passthrough'}

step = ModelingStep(context)
step.run()
