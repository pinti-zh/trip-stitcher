import numpy as np
from loguru import logger
from ocsept.data.generation.speed.time_optimal import TimeOptimalStrategy
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.itinerary import TravelItinerary
from ocsept.models.transport.mission import DrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics
from optool.optimization.helpers import UnsuccessfulOptimization
from optool.uom import Quantity
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from models import EnergyDemand
from functools import partial
from pipeline_utils import get_default_parser, run_pipeline, setup_logger, suppress_stdout
from vehicles import mini, maxi, mega


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


def main(itinerary: TravelItinerary, bus_type=None) -> EnergyDemand:
    bus = None
    match bus_type:
        case "mini":
            bus = mini.bus
        case "maxi":
            bus = maxi.bus
        case "mega":
            bus = mega.bus
        case _:
            raise NotImplementedError(f"Unknown bus type: {bus_type}")
    assert bus is not None
    comfort = RidingComfort()

    logger.debug("Calculating speed profile")

    itinerary = smooth_elevation(itinerary)

    with suppress_stdout():
        try:
            speed_profile = TimeOptimalStrategy().process(itinerary, bus, comfort)
        except UnsuccessfulOptimization:
            logger.error("not ok")
            return EnergyDemand(magnitude=0, units="kWh", time=0)

    logger.debug("Ok")

    mission = DrivingMission(
        name="Fastest possible travel speed",
        time=speed_profile.time,
        speed=speed_profile.speed,
        inclination=itinerary.get_inclination(speed_profile.distance),
        payload="0 kg",
    )

    with suppress_stdout():
        x = LongitudinalVehicleDynamics.of(bus, mission)

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
    return EnergyDemand(magnitude=total_energy.magnitude, units=str(total_energy.units), time=mission.time[-1].m_as("s"))


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--bus-type", choices=["mini", "maxi", "mega"], default="maxi", help="bus type")
    args = parser.parse_args()
    setup_logger(args.debug)
    run_pipeline(partial(main, bus_type=args.bus_type), TravelItinerary, args)
