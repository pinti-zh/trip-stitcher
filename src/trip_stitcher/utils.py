import os
import sys
from contextlib import contextmanager
from datetime import datetime

from loguru import logger


def str_to_datetime(value: str) -> datetime:
    hour, minute, second = [int(v) for v in value.split(":")]
    if hour >= 24:
        return datetime(year=2025, month=1, day=14, hour=hour - 24, minute=minute, second=second)
    else:
        return datetime(year=2025, month=1, day=13, hour=hour, minute=minute, second=second)


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


def setup_logger(debug: bool):
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
