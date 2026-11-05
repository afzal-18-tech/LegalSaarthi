"""
LegalSaarthi logging setup.

Provides structured logging with file and console output.
"""
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    config: dict[str, Any] | None = None,
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the root LegalSaarthi logger.

    Args:
        config: Optional config dict (uses logging section).
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    if config and "logging" in config:
        log_cfg = config["logging"]
        level = log_cfg.get("level", level)
        fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        if log_file is None:
            log_file = log_cfg.get("file")
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logger = logging.getLogger("legalsaarthi")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler — force utf-8 on Windows to handle non-ASCII metadata
    stdout = sys.stdout
    if hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    console = logging.StreamHandler(stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the legalsaarthi namespace.

    Args:
        name: Logger name (e.g. 'ingestion.metadata').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"legalsaarthi.{name}")
