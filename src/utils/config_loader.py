"""
utils/config_loader.py — YAML Configuration Loader for the Credit Risk Pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {path}\n"
            f"Localização esperada: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:

            raise yaml.YAMLError(f"Erro de sintaxe no YAML {path}: {exc}")from exc
