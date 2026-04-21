"""
utils/config_loader.py — Carregador de Configurações YAML.

Este módulo centraliza a leitura de contratos (YAML) para garantir que
mudanças na estrutura de pastas não exijam alterações na lógica de negócio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Carrega um arquivo YAML e o converte em dicionário Python.

    Args:
        path: Caminho absoluto ou relativo para o arquivo .yaml.

    Returns:
        Dicionário com as configurações do arquivo. Retorna {} se vazio.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        yaml.YAMLError: Se o arquivo estiver com sintaxe inválida.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {path}\n"
            f"Localização esperada: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            # Em backend, queremos saber exatamente onde o erro ocorreu
            raise yaml.YAMLError(f"Erro de sintaxe no YAML {path}: {exc}") from exc
