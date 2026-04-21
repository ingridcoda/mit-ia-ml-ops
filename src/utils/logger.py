"""
utils/logger.py — Fábrica de Logs do Pipeline de Crédito.

Garante que todas as etapas (EDA, Ingestão, Modelagem) usem o mesmo formato
de saída e nível de verbosidade, facilitando o debug em produção.
"""
import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, logging_config: dict[str, Any] = None) -> logging.Logger:
    """
    Constrói e retorna um logger configurado.

    Se chamado múltiplas vezes com o mesmo nome, retorna a mesma instância,
    evitando duplicação de mensagens.
    """
    if logging_config is None:
        logging_config = {}

    logger = logging.getLogger(name)

    # Evita adicionar handlers se o logger já estiver configurado
    if logger.handlers:
        return logger

    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # Formato profissional de log (Data - Nível - Módulo - Mensagem)
    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    datefmt = logging_config.get("datefmt", "%H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Handler para o Console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler opcional para arquivo (Auditoria)
    if logging_config.get("log_to_file", False):
        log_file = logging_config.get("log_file", "credit_pipeline.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Impede propagação para o root logger (evita logs duplicados no Streamlit)
    logger.propagate = False

    return logger
