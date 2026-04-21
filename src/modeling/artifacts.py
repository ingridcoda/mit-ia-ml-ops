"""
modeling/artifacts.py — Gerador de artefatos visuais (Matriz de Confusão/ROC).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay


class ArtifactGenerator:
    def __init__(self, artifacts_cfg: dict, logger: logging.Logger | None = None) -> None:
        self.artifacts_cfg = artifacts_cfg
        self.output_dir = Path(artifacts_cfg.get('output_dir', 'outputs/modeling'))
        self.logger = logger

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusão: {model_name}")
        path = self.output_dir / f"cm_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_roc_curve(self, y_true, y_prob, model_name):
        if y_prob is None: return None
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
        path = self.output_dir / f"roc_{model_name}.png"
        fig.savefig(path)
        plt.close(fig)
        return path
