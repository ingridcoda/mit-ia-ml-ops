"""
test_base.py — Validation of the system's contracts (ABCs).
"""
import pytest

from src.core.base import PipelineStep, DataLoader


def test_pipeline_step_instanciacao_abstrata():
    """Garante que não se pode instanciar o contrato diretamente."""
    with pytest.raises(TypeError):
        PipelineStep(logger=None)


def test_pipeline_step_implementacao_obrigatoria():
    """Valida que subclasses devem implementar o método run."""

    class StepIncompleto(PipelineStep):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        StepIncompleto(logger=None)


def test_data_loader_contrato_load():
    """Garante que o DataLoader exige o método load."""

    class LoaderIncompleto(DataLoader):
        pass

    with pytest.raises(TypeError):
        LoaderIncompleto(logger=None)
