"""Thin logging wrapper for scene_mining.

Provides get_logger() and log_dict() that delegate to mmengine.logging.MMLogger
when available, falling back to the standard library otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from mmengine.logging import MMLogger as _MMLogger

    def get_logger(name: str = "scene_mining") -> _MMLogger:
        """Return an MMLogger instance for *name*.

        The first call with a given name creates the logger; subsequent
        calls return the cached instance.
        """
        try:
            return _MMLogger.get_instance(name)
        except Exception:
            # Logger may already exist — fetch it by name
            return _MMLogger.get_current_instance()

    def log_dict(d: dict[str, Any], level: str = "INFO", logger_name: str = "scene_mining") -> None:
        """Log each key=value pair in *d* at the specified level."""
        log = get_logger(logger_name)
        lvl = getattr(logging, level.upper(), logging.INFO)
        for k, v in d.items():
            log.log(lvl, "%s = %s", k, v)

except ImportError:
    # Fallback: plain stdlib logging
    def get_logger(name: str = "scene_mining") -> logging.Logger:  # type: ignore[misc]
        """Return a stdlib Logger (mmengine not installed)."""
        log = logging.getLogger(name)
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
        return log

    def log_dict(d: dict[str, Any], level: str = "INFO", logger_name: str = "scene_mining") -> None:  # type: ignore[misc]
        """Log each key=value pair in *d* at the specified level."""
        log = get_logger(logger_name)
        lvl = getattr(logging, level.upper(), logging.INFO)
        for k, v in d.items():
            log.log(lvl, "%s = %s", k, v)


__all__ = ["get_logger", "log_dict"]
