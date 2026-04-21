"""
test_artifacts.py — Testes para o gerador de artefatos visuais (Matriz de Confusão/ROC).
"""
import numpy as np
import pytest

from src.modeling.artifacts import ArtifactGenerator


@pytest.fixture
def generator(tmp_path, null_logger):
    """Instancia o gerador apontando para uma pasta temporária."""
    cfg = {'output_dir': str(tmp_path)}
    return ArtifactGenerator(cfg, logger=null_logger)


def test_plot_confusion_matrix_criacao_arquivo(generator, tmp_path):
    """Verifica se o arquivo da matriz de confusão é gerado corretamente."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    path = generator.plot_confusion_matrix(y_true, y_pred, "test_model")

    assert path.exists()
    assert path.suffix == ".png"
    assert "cm_test_model.png" in str(path)


def test_plot_roc_curve_criacao_arquivo(generator, tmp_path):
    """Verifica se o arquivo da curva ROC é gerado quando há probabilidades."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])

    path = generator.plot_roc_curve(y_true, y_prob, "test_model")

    assert path.exists()
    assert "roc_test_model.png" in str(path)


def test_plot_roc_curve_ignora_sem_probabilidade(generator):
    """Garante que a curva ROC retorna None se não houver probabilidades."""
    path = generator.plot_roc_curve([0, 1], None, "test_model")
    assert path is None
