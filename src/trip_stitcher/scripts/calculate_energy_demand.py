from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from ocsept.data.generation.speed.time_optimal import TimeOptimalStrategy
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.itinerary import TravelItinerary
from ocsept.models.transport.mission import DrivingMission as OcseptDrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics
from optool.optimization.helpers import UnsuccessfulOptimization
from optool.uom import Quantity
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from trip_stitcher.models import Stop, Trip
from trip_stitcher.osrm import create_itinerary_from_trip
from trip_stitcher.pipeline import get_default_parser, run_pipeline
from trip_stitcher.utils import setup_logger, suppress_stdout
from trip_stitcher.vehicles import maxi, mega, mini


def set_dark_mode(fig, ax):
    ax.set_xlabel("X-axis", color="white")
    ax.set_ylabel("Y-axis", color="white")
    ax.tick_params(colors="white", which="both")

    # Optional: make grid slightly transparent white
    ax.grid(color="white", alpha=0.2)
    # Make the figure and axes background transparent
    fig.patch.set_alpha(0)  # figure background
    ax.patch.set_alpha(0)  # axes background


def smooth_elevation(itinerary: TravelItinerary, dx: float = 1.0, window_length: int = 401) -> TravelItinerary:
    elevation = itinerary.elevation_profile.to("meters").magnitude
    distance_as_quantity = itinerary.cumulative_distances
    inclination = itinerary.get_inclination(distance_as_quantity)
    logger.debug(f"Inclination ranging from {min(inclination)} to {max(inclination)}")

    distance = distance_as_quantity.to("meters").magnitude

    distance_uniform = np.arange(distance[0], distance[-1], dx)
    interp_func = interp1d(distance, elevation, kind="linear")  # or 'cubic'
    elevation_uniform = interp_func(distance_uniform)

    order = 2

    elevation_uniform_smooth = savgol_filter(elevation_uniform, window_length, order)
    interp_back = interp1d(distance_uniform, elevation_uniform_smooth, kind="linear")
    elevation_smoothed = interp_back(distance[:-1])

    for e, waypoint in zip(elevation_smoothed, itinerary.waypoints[:-1]):
        waypoint.elevation = f"{e} m"

    inclination = itinerary.get_inclination(distance_as_quantity)
    logger.debug(f"Inclination ranging from {min(inclination)} to {max(inclination)} after smoothing")

    return itinerary


def calculate_energy_demand(trip: Trip, bus_type=None, plot=False, df: pd.DataFrame | None = None) -> Trip:
    if df is None:
        raise ValueError("df must not be None")
    stop_dict = dict((stop.id, stop) for stop in Stop.list_from_dataframe(df))
    itinerary = create_itinerary_from_trip(trip, stop_dict)

    bus = None
    match bus_type:
        case "mini":
            bus = mini.bus
        case "maxi":
            bus = maxi.bus
        case "mega":
            bus = mega.bus
        case _:
            raise ValueError(f"Unknown bus type: {bus_type}")
    assert bus is not None
    comfort = RidingComfort()

    logger.debug("Calculating speed profile")

    itinerary = smooth_elevation(itinerary)

    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        set_dark_mode(fig, ax)
        ax.plot(itinerary.cumulative_distances, itinerary.elevation_profile, c="navy")
        plt.title("Elevation Profile", color="white")
        plt.xlabel("Distance [m]")
        plt.ylabel("Elevation [m]")
        plt.show()

    with suppress_stdout():
        try:
            speed_profile = TimeOptimalStrategy().process(itinerary, bus, comfort)
        except UnsuccessfulOptimization:
            logger.error("Unsuccessful optimization while calculating speed profile")
            return trip

    assert speed_profile.distance is not None
    assert speed_profile.speed is not None
    assert speed_profile.time is not None

    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        set_dark_mode(fig, ax)
        ax.plot(speed_profile.distance, speed_profile.speed, c="goldenrod")
        for speed_limit in [30, 50, 60, 80]:
            ax.plot(
                speed_profile.distance,
                [speed_limit / 3.6] * len(speed_profile.distance),
                label=f"{speed_limit} m/s",
                linestyle="dashed",
            )
        plt.title("Speed Profile", color="white")
        plt.xlabel("Distance [m]")
        plt.ylabel("Speed [m/s]")
        plt.legend()
        plt.show(block=False)

    logger.debug("Ok")

    mission = OcseptDrivingMission(
        name="Fastest possible travel speed",
        time=speed_profile.time,
        speed=speed_profile.speed,
        inclination=itinerary.get_inclination(speed_profile.distance),
        payload="0 kg",
    )

    with suppress_stdout():
        x = LongitudinalVehicleDynamics.of(bus, mission)

    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        set_dark_mode(fig, ax)
        plt.plot(speed_profile.distance, x.traction_force, c="firebrick")
        plt.title("Traction Force", color="white")
        plt.xlabel("Distance [m]")
        plt.ylabel("Force [kN]")
        plt.show(block=False)

    dt = [t2 - t1 for t1, t2 in zip(speed_profile.time[:-1], speed_profile.time[1:])]
    average_velocity = [(v1 + v2) / 2 for v1, v2 in zip(speed_profile.speed[:-1], speed_profile.speed[1:])]
    efficiency = 0.9
    propulsion_power = []
    for f, v in zip(x.traction_force[:-1], average_velocity):
        p_mech = f * v
        if p_mech >= 0:
            propulsion_power.append(p_mech / efficiency)
        else:
            propulsion_power.append(p_mech * efficiency)
    aux_power = Quantity(0, "kW")
    total_power = [p + aux_power for p in propulsion_power]
    total_energy = Quantity(0, "kWh")
    for t, p in zip(dt, total_power):
        total_energy += t * p

    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        set_dark_mode(fig, ax)
        plt.plot(
            [t.magnitude for t in speed_profile.time[:-1]], [p.magnitude for p in propulsion_power], c="forestgreen"
        )
        plt.title("Power", color="white")
        plt.xlabel("Time [s]")
        plt.ylabel("Power [kW]")
        plt.show()

    assert str(total_energy.units) == "kWh"
    trip.estimated_energy_demand = total_energy.magnitude * 3.6e6
    return trip


def main():
    parser = get_default_parser()
    parser.add_argument("--bus-type", choices=["mini", "maxi", "mega"], default="maxi", help="bus type")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--plot", action="store_true", help="create plots of trips and results")
    args = parser.parse_args()
    df = pd.read_parquet(args.data_file)
    setup_logger(args.debug)
    run_pipeline(partial(calculate_energy_demand, bus_type=args.bus_type, plot=args.plot, df=df), Trip, args)


if __name__ == "__main__":
    main()
