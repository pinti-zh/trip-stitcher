from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from trip_stitcher.models import DrivingMission, Stop, Trip, str_to_datetime
from trip_stitcher.pipeline import collect, get_default_parser
from trip_stitcher.plot_utils import color_map, name_to_line_map
from trip_stitcher.stitching import (
    driving_mission_ends_at_trip_start,
    driving_mission_ends_before_trip,
    stitch_trips_into_driving_missions,
)
from trip_stitcher.utils import setup_logger


def stitch(trips: list[Trip], args: Namespace, df: pd.DataFrame | None = None) -> list[DrivingMission]:
    if df is None:
        raise ValueError("df must not be None")

    stop_dict = dict((stop.id, stop) for stop in Stop.list_from_dataframe(df))

    driving_missions = stitch_trips_into_driving_missions(
        trips, lambda dm, t: driving_mission_ends_at_trip_start(dm, t) and driving_mission_ends_before_trip(dm, t)
    )

    logger.debug(f"Distributed {len(trips)} trips to {len(driving_missions)} driving missions")

    if args.plot:
        terminal_stops = []
        for trip in trips:
            terminal_stops.append(stop_dict[trip.stops[0]])
            terminal_stops.append(stop_dict[trip.stops[-1]])

        line_map = name_to_line_map(terminal_stops)

        sns.set_style("darkgrid")
        colors = [color_map(i / len(driving_missions)) for i in range(len(driving_missions))]
        for color, driving_mission in zip(colors, driving_missions):
            for trip in driving_mission.trips:
                trip_start_time = str_to_datetime(trip.arrival_times[0])
                trip_end_time = str_to_datetime(trip.arrival_times[-1])
                start = line_map[stop_dict[trip.stops[0]].name]
                stop = line_map[stop_dict[trip.stops[-1]].name]
                plt.plot([trip_start_time, trip_end_time], [start, stop], c=color, marker=".", markersize=5)
        plt.yticks(list(line_map.values()), list(line_map.keys()))
        plt.title("Driving Missions")
        plt.show()
    return driving_missions


def main():
    parser = get_default_parser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    setup_logger(args.debug)

    trips = collect(Trip, args)
    logger.debug(f"Collected {len(trips)} trips")
    df = pd.read_parquet(args.data_file)
    stitch(trips, args, df=df)


if __name__ == "__main__":
    main()
