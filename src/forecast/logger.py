"""
FORECAST logging utilities.

Provides :data:`LOGGER` as the root console logger and :func:`get_logger`
for obtaining named, file-backed loggers for each pipeline step.

Per-step log files
------------------
Each CLI step should call :func:`get_logger` with its step name and the
``--log-dir`` path::

    from forecast.logger import get_logger
    log = get_logger("collect", log_dir=Path("logs/"), instance_id="astropy__astropy-12907")
    log.info("Starting instrumentation…")

This creates ``logs/collect_astropy__astropy-12907.log`` in addition to the
console handler.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Root logger (console only, used by internal modules)
# ─────────────────────────────────────────────────────────────────────────────

LOGGER = logging.getLogger("forecast")

if not LOGGER.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.DEBUG)
    _console.setFormatter(
        logging.Formatter("%(name)s :: %(levelname)-8s :: %(message)s")
    )
    LOGGER.addHandler(_console)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

_LOG_FMT = "%(levelname)-8s  %(name)s  %(message)s"

# ─────────────────────────────────────────────────────────────────────────────
# Per-step logger factory
# ─────────────────────────────────────────────────────────────────────────────


def get_logger(
    step: str,
    log_dir: Optional[Path] = None,
    instance_id: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Return a :class:`logging.Logger` for the given pipeline *step*.

    If *log_dir* is provided a :class:`~logging.FileHandler` is attached
    that writes to ``<log_dir>/<step>[_<instance_id>].log``.  A
    :class:`~logging.StreamHandler` to *stdout* is always attached.

    Calling :func:`get_logger` twice with the same *step* / *instance_id*
    returns the **same** logger without duplicating handlers.

    :param step: Pipeline step name used as the logger name and log file stem
        (e.g. ``"collect"``, ``"tree"``, ``"diagnose"``, ``"prompt"``,
        ``"run"``, ``"evaluate"``).
    :param log_dir: Directory for the log file.  ``None`` disables file
        logging.
    :param instance_id: Subject identifier appended to the file name for
        disambiguation.  ``None`` omits the suffix.
    :param level: Logging level for the file handler.  Defaults to
        :data:`logging.DEBUG`.
    :returns: A configured :class:`logging.Logger` instance.
    """
    suffix = f".{instance_id}" if instance_id else ""
    logger_name = f"forecast.{step}{suffix}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(_LOG_FMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = step if not instance_id else f"{step}_{instance_id}"
        fh = logging.FileHandler(log_dir / f"{stem}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
