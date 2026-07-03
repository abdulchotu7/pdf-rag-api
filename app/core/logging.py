import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    logger = logging.getLogger("rag")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rag.{name}")
