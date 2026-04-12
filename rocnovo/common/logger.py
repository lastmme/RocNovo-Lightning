import sys
import logging
from pathlib import Path

from loguru import logger

from rocnovo.common.io import normalize_path

logger.remove()
logger.add(sys.stderr)

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        is_lightning_log = record.name.startswith(("lightning", "pytorch_lightning"))
        
        if is_lightning_log and level == "INFO":
            level = "DEBUG"

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def redirect_lightning_logs_to_loguru():
    logging.captureWarnings(True)
    target_loggers = (
        "pytorch_lightning", 
        "lightning.pytorch", 
        "lightning.fabric",
        "lightning_fabric",
        "torch.distributed",
        "py.warnings"
    )
    
    for name in target_loggers:
        pl_logger = logging.getLogger(name)
        pl_logger.setLevel(logging.INFO)
        pl_logger.handlers.clear() 
        pl_logger.addHandler(InterceptHandler())
        pl_logger.propagate = False

def set_logger_dir(dir_path: str | Path):
    logger_dir = normalize_path(dir_path) / "logs"
    logger_dir.mkdir(parents=True, exist_ok=True)
    logger.add(logger_dir / "info.log", filter=lambda record: record["level"].name == "INFO")
    logger.add(logger_dir / "debug.log", filter=lambda record: record["level"].name == "DEBUG")
    logger.add(logger_dir / "error.log", level="ERROR")