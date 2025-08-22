"""
Logger utility for consistent logging across the project.
"""
import logging
from typing import Optional

def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """
    Get a configured logger instance.
    Args:
        name: Logger name (None for root logger)
        level: Logging level (default: logging.INFO)
        log_file: Optional file path for file logging
        fmt: Log message format
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # File handler (optional)
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger 