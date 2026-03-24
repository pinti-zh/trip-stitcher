import numpy as np
import pandas as pd
from ocsept.data.generation.speed.time_optimal import TimeOptimalStrategy
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.mission import DrivingMission as OcseptDrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics
from optool.uom import Quantity

from trip_stitcher.elevation import ElevationOracle
from trip_stitcher.models import Stop, Trip
from trip_stitcher.utils import str_to_datetime, suppress_stdout
from trip_stitcher.vehicles import maxi, mega, mini


class EnergyDemandEstimator:
    def __init__(self, df: pd.DataFrame):
        self.stop_dict: dict[str, Stop] = dict((stop.id, stop) for stop in Stop.list_from_dataframe(df))
        self.elevation_oracle = ElevationOracle()
        self.cache: dict[tuple[str, ...], float] = {}

    def calculate_energy_demand(self, trip: Trip, bus_type: str | None = None, aux_power: float = 0.0) -> float:
        cache_key = tuple(trip.stops + [str(aux_power), str(bus_type)])
        if cache_key in self.cache.keys():
            return self.cache[cache_key]

        trip_geometry = trip.download_geometry(self.stop_dict, elevation_oracle=self.elevation_oracle)

        max_inclination = max(
            abs(100 * (e2 - e1) / d)
            for d, e1, e2 in zip(trip_geometry.distance, trip_geometry.elevation[:-1], trip_geometry.elevation[1:])
            if d > 0
        )
        assert max_inclination < 20

        itinerary = trip_geometry.create_itinerary()

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

        with suppress_stdout():
            speed_profile = TimeOptimalStrategy().process(itinerary, bus, comfort)

        assert speed_profile.distance is not None
        assert speed_profile.speed is not None
        assert speed_profile.time is not None

        mission = OcseptDrivingMission(
            name="Fastest possible travel speed",
            time=speed_profile.time,
            speed=speed_profile.speed,
            inclination=itinerary.get_inclination(speed_profile.distance),
            payload="0 kg",
        )

        with suppress_stdout():
            vehicle_dynamics = LongitudinalVehicleDynamics.of(bus, mission)

        average_velocity = speed_profile.speed[:-1] + speed_profile.speed[1:] / 2
        p_mech = average_velocity * vehicle_dynamics.traction_force[:-1]
        efficiency = 0.9
        propulsion_power = np.where(p_mech >= 0, p_mech / efficiency, p_mech * efficiency)
        propulsion_energy = np.sum(np.diff(speed_profile.time) * propulsion_power).to("J")

        trip_duration = str_to_datetime(trip.arrival_times[-1]) - str_to_datetime(trip.arrival_times[0])
        aux_energy = (Quantity(aux_power, "W") * Quantity(trip_duration.total_seconds(), "s")).to("J")

        total_energy = propulsion_energy.magnitude + aux_energy.magnitude
        self.cache[cache_key] = total_energy
        return total_energy
