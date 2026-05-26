# find_trips.py
#
# Pipeline script that reads RadiusQuery objects from a JSONL input stream,
# searches the RouteLocator for all routes and trips within the specified
# radius, and writes the matching Trip objects (sorted by first arrival time)
# to a JSONL output stream.
#
# The RouteLocator index is built once at startup from a Parquet data file.
#
# Usage:
#   python find_trips.py --data-file FILE
#                        [--input FILE] [--output FILE] [--debug]
#
# Arguments:
#   --data-file   Path to the Parquet file containing route/trip/stop data
#   --debug       Enable verbose DEBUG-level logging

from functools import partial
from time import perf_counter

import pandas as pd
from loguru import logger

from trip_stitcher.models import RadiusQuery, Route, Stop, Trip
from trip_stitcher.pipeline import get_default_parser, run_pipeline
from trip_stitcher.route_locator import RouteLocator
from trip_stitcher.utils import setup_logger


def find_trips(
    radius_query: RadiusQuery,
    route_locator: RouteLocator | None = None,
    df: pd.DataFrame | None = None,
) -> list[Trip]:
    if route_locator is None:
        raise ValueError("route_locator must be provided")
    if df is None:
        raise ValueError("df must be provided")

    routes, _ = route_locator.radius_query(radius_query)
    logger.debug(
        f"{len(routes)} routes found within {radius_query.radius} meters of ({radius_query.lat}, {radius_query.lon})"
    )
    first_few_enrty_string = ", ".join(route.name for route in routes[:3]) + ", ..." * (
        len(routes) > 3
    )
    logger.debug(f"Routes: {first_few_enrty_string}")

    trips: list[Trip] = []
    for route in routes:
        trips += [route_locator._trip_dict[trip_id] for trip_id in route.trips]
    logger.debug(f"{len(trips)} trips found")
    return list(sorted(trips, key=lambda t: t.arrival_times[0]))


def main():
    parser = get_default_parser()
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()
    setup_logger(args.debug)

    df = pd.read_parquet(args.data_file)
    start_time = perf_counter()
    route_locator = RouteLocator(
        Route.list_from_dataframe(df), Trip.list_from_dataframe(df), Stop.list_from_dataframe(df)
    )
    logger.debug(f"Built route locator in {perf_counter() - start_time:.4f}s")

    run_pipeline(partial(find_trips, route_locator=route_locator, df=df), RadiusQuery, args)


if __name__ == "__main__":
    main()
