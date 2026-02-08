"""Structured logging helpers using the stdlib + Rich."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with Rich handler (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring Rich logging is set up."""
    setup_logging()
    return logging.getLogger(name)
