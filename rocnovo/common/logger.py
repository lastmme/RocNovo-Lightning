import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr)

def set_logger_dir(dir_path: str | Path):
    logger_dir = dir_path / "logs"
    logger_dir.mkdir(parents=True, exist_ok=True)
    logger.add(logger_dir / "info.log", filter=lambda record: record["level"].name == "INFO")
    logger.add(logger_dir / "debug.log", filter=lambda record: record["level"].name == "DEBUG")
    logger.add(logger_dir / "error.log", level="ERROR")