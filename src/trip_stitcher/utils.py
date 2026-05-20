import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
from loguru import logger
from skimage.restoration import denoise_tv_chambolle
from sqlalchemy import Engine, MetaData, select
import Levenshtein


def str_to_datetime(value: str, start: datetime = datetime(year=2025, month=1, day=1)) -> datetime:
    hour, minute, second = [int(v) for v in value.split(":")]
    if hour >= 24:
        next_day = start + timedelta(days=1)
        return datetime(
            year=next_day.year,
            month=next_day.month,
            day=next_day.day,
            hour=hour - 24,
            minute=minute,
            second=second,
        )
    else:
        return datetime(
            year=start.year,
            month=start.month,
            day=start.day,
            hour=hour,
            minute=minute,
            second=second,
        )


def datetime_to_str(dt: datetime, is_next_day: bool = False) -> str:
    datetime_str = dt.strftime("%H:%M:%S")
    if is_next_day:
        datetime_str = (
            str(int(datetime_str.split(":")[0]) + 24) + ":" + ":".join(datetime_str.split(":")[1:])
        )
    return datetime_str


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


def match_agency_string(query: str, engine: Engine, metadata: MetaData) -> tuple[str, str]:
    agency = metadata.tables["agency"]
    agency_stmt = select(agency.c.agency_id, agency.c.agency_name)
    agency_id_to_name = {}
    agency_name_to_id = {}
    with engine.connect() as conn:
        for result in conn.execute(agency_stmt):
            agency_id_to_name[result.agency_id] = result.agency_name
            agency_name_to_id[result.agency_name] = result.agency_id

    closest_agency_name = sorted(
        agency_name_to_id.keys(),
        key=lambda x: Levenshtein.distance(x.lower(), query.lower()) / len(x),
    )[0]
    closest_agency_id = sorted(
        agency_id_to_name.keys(),
        key=lambda x: Levenshtein.distance(x.lower(), query.lower()) / len(x),
    )[0]

    name_distance = Levenshtein.distance(closest_agency_name.lower(), query.lower()) / len(
        closest_agency_name
    )
    id_distance = Levenshtein.distance(closest_agency_id.lower(), query.lower()) / len(
        closest_agency_id
    )

    if name_distance < id_distance:
        agency_id = agency_name_to_id[closest_agency_name]
    else:
        agency_id = closest_agency_id

    return agency_id, agency_id_to_name[agency_id]
