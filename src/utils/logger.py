"""
logger.py: Utility function to set up logging for the credit pipeline project.
"""

import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, logging_config: dict[str, Any] = None) -> logging.Logger:
    if logging_config is None:
        logging_config = {}

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    datefmt = logging_config.get("datefmt", "%H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if logging_config.get("log_to_file", False):
        log_file = logging_config.get("log_file", "credit_pipeline.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
