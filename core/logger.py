# core/logger.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralised logger for the pipeline. Works both in Databricks notebooks
# (where print() goes to stdout) and in .py modules.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
from config.config import LOG_LEVEL


def get_logger(name: str = "experian_feature_store") -> logging.Logger:
    """
    Returns a configured logger. Call this at the top of every module:

        from core.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
