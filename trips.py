import sys
import os
from argparse import ArgumentParser, FileType
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from models import Trip


def gtfs_date_string_to_date(s: str) -> date:
    year = s[:4]
    month = int(s[4:6])
    day = int(s[6:8])
    return date(year=int(year), month=month, day=day)


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output", type=FileType("w"), default=sys.stdout, help="Output file (defaults to stdout)")
    args = parser.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    file_path = Path(args.file)
    df = pd.read_parquet(file_path)

    day = date(year=2025, month=1, day=20)

    relevant_routes = [
        "96-241-7-j25-1",  # 501
        "96-241-7-j26-1",  # 501
        "96-241-8-j25-1",  # 502
        "96-241-8-j26-1",  # 502
        "96-241-5-j25-1",  # 503
        "96-241-5-j26-1",  # 503
    ]

    filtered_df = df[df["route_id"].isin(relevant_routes)]
    filtered_df = filtered_df[filtered_df["monday"] == 1]  # Monday - Friday
    contains_date = []
    for start_date_str, end_date_str in zip(filtered_df["start_date"], filtered_df["end_date"]):
        start_date = gtfs_date_string_to_date(start_date_str)
        end_date = gtfs_date_string_to_date(end_date_str)
        contains_date.append(start_date <= day <= end_date)

    filtered_df["contains_date"] = contains_date
    filtered_df = filtered_df[filtered_df["contains_date"]]

    # filtered_df = filtered_df[filtered_df["route_short_name"] == "501"]
    filtered_df = filtered_df.sort_values(by=["arrival_time", "stop_sequence"])

    trip_dict = {}

    for route_name, stop_sequence, stop_name, arrival_time, trip, lon, lat in zip(
        filtered_df["route_short_name"],
        filtered_df["stop_sequence"],
        filtered_df["stop_name"],
        filtered_df["arrival_time"],
        filtered_df["trip_id"],
        filtered_df["stop_lon"],
        filtered_df["stop_lat"],
    ):
        if trip not in trip_dict.keys():
            trip_dict[trip] = {
                "route": route_name,
                "stops": [stop_name],
                "stop_sequence": [stop_sequence],
                "arrival_times": [arrival_time],
                "lon": [lon],
                "lat": [lat]
            }
        else:
            assert trip_dict[trip]["route"] == route_name
            assert trip_dict[trip]["stop_sequence"][-1] + 1 == stop_sequence
            trip_dict[trip]["stops"].append(stop_name)
            trip_dict[trip]["stop_sequence"].append(stop_sequence)
            trip_dict[trip]["arrival_times"].append(arrival_time)
            trip_dict[trip]["lon"].append(lon)
            trip_dict[trip]["lat"].append(lat)

    trips = [Trip(name=key, **values) for key, values in trip_dict.items()]
    logger.debug(f"Created {len(trips)} trips")
    try:
        for trip in trips:
            print(trip.json(), file=args.output, flush=True)
    except BrokenPipeError:
        # downstream command closed the pipe (e.g. head)
        os._exit(0)


if __name__ == "__main__":
    main()
