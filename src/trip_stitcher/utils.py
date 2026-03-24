import os
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from loguru import logger
from skimage.restoration import denoise_tv_chambolle


def str_to_datetime(value: str) -> datetime:
    hour, minute, second = [int(v) for v in value.split(":")]
    if hour >= 24:
        return datetime(year=2025, month=1, day=14, hour=hour - 24, minute=minute, second=second)
    else:
        return datetime(year=2025, month=1, day=13, hour=hour, minute=minute, second=second)


def limits_from_speed(values: list[float]) -> list[float]:
    denoised_values = denoise_tv_chambolle(np.array(values), weight=1.0)
    dv = np.abs(np.diff(denoised_values))
    breaks = np.where(dv > 10.0)[0]
    segments = np.split(denoised_values, breaks + 1)
    piece_wise_speed = np.concatenate([np.full_like(seg, seg.mean()) for seg in segments])
    return [snap_to_limits(v) for v in piece_wise_speed]


def snap_to_limits(x: float) -> float:
    for limit in [30.0, 50.0, 60.0]:
        if limit > x:
            return limit
    return 80.0


@contextmanager
def suppress_stdout(supress_stderr=False):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            if supress_stderr:
                sys.stderr = devnull
            yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def setup_logger(debug: bool):
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def upsample(x: list[float], samples: list[float], has_sample: list[bool]) -> list[float]:
    assert len(x) == len(has_sample)
    assert len([value for value in has_sample if value]) == len(samples)
    assert has_sample[0] and has_sample[-1]

    up_sampled = np.interp(x, [v for v, hs in zip(x, has_sample) if hs], samples)
    return up_sampled.tolist()
