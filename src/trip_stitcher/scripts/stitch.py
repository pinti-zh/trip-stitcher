from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from trip_stitcher.models import DrivingMission, Stop, Trip, str_to_datetime
from trip_stitcher.pipeline import collect, get_default_parser
from trip_stitcher.utils import setup_logger


def peak(value: float) -> float:
    assert 0 <= value <= 1
    if 1 / 6 <= value <= 3 / 6:
        return 1
    else:
        d1 = abs(1 / 6 - value)
        d2 = abs(3 / 6 - value)
        return max(1 - 6 * d1, 1 - 6 * d2, 0)


def color_map(value: float) -> tuple[float, float, float]:
    value = max(min(value, 1), 0)
    red_value = value + 1 / 3
    if red_value > 1:
        red_value -= 1
    green_value = value - 1 / 3
    if green_value < 0:
        green_value += 1
    blue_value = value
    return peak(red_value), peak(green_value), peak(blue_value)


def stitch(trips: list[Trip], args: Namespace, df: pd.DataFrame | None = None) -> list[DrivingMission]:
    if df is None:
        raise ValueError("df must not be None")
    stop_names = dict((stop.id, stop.name) for stop in Stop.list_from_dataframe(df))

    driving_missions = []
    unique_stops = []
    for trip in trips:
        unique_stops.append(stop_names[trip.stops[0]].split(":")[0])
        unique_stops.append(stop_names[trip.stops[-1]].split(":")[0])
        driving_missions = sorted(driving_missions, key=lambda m: m.end_time)
        for driving_mission in filter(lambda m: m.is_addable(trip), driving_missions):
            driving_mission.add_trip(trip)
            break
        else:
            new_driving_mission = DrivingMission()
            new_driving_mission.add_trip(trip)
            driving_missions.append(new_driving_mission)

    logger.debug(f"Distributed {len(trips)} trips to {len(driving_missions)} driving missions")
    unique_stops = list(set(unique_stops))
    stops_to_int = dict((stop_name, i + 1) for i, stop_name in enumerate(unique_stops))

    if args.plot:
        sns.set_style("darkgrid")
        colors = [color_map(i / len(driving_missions)) for i in range(len(driving_missions))]
        for color, driving_mission in zip(colors, driving_missions):
            for trip in driving_mission.trips:
                trip_start_time = str_to_datetime(trip.arrival_times[0])
                trip_end_time = str_to_datetime(trip.arrival_times[-1])
                start = stops_to_int[stop_names[trip.stops[0]].split(":")[0]]
                stop = stops_to_int[stop_names[trip.stops[-1]].split(":")[0]]
                plt.plot([trip_start_time, trip_end_time], [start, stop], c=color, marker=".", markersize=5)
        plt.yticks(list(stops_to_int.values()), list(stops_to_int.keys()))
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
