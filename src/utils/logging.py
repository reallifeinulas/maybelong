"""Loguru tabanlı yapılandırılmış logger yardımcıları.

Örnek:
    from src.utils.logging import setup_logger

    logger = setup_logger()
    logger.info("Pipeline başlatıldı")
"""

from __future__ import annotations

from loguru import logger


def setup_logger(level: str = "INFO"):
    """Loguru logger'ını standart formatla kur."""

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        colorize=True,
    )
    return logger
