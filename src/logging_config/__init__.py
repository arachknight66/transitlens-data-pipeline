"""Application logging configuration."""

import logging
import sys

from loguru import logger


def configure_logging(level: str) -> None:
    """Configure structured application logging for the current process.

    Args:
        level: Minimum logging level accepted by Loguru.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        serialize=True,
        backtrace=False,
        diagnose=False,
    )
    logging.captureWarnings(True)
