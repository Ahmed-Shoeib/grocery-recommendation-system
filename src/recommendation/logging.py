"""Centralized logging setup.

Call `setup_logging()` once at process start (API startup, dashboard start,
training script entrypoint); use `get_logger(__name__)` everywhere else.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

# Third-party libraries that log routine HTTP/IO chatter at INFO (notably
# huggingface_hub's connectivity check on every SentenceTransformer load in
# Phase 3+) - quieted so application logs stay legible. Not silenced
# entirely: WARNING and above still surface.
_NOISY_THIRD_PARTY_LOGGERS = ["httpx", "httpcore", "urllib3", "filelock"]


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
