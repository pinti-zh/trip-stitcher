from contextlib import contextmanager

import numpy as np
from IPython.utils.io import capture_output
from loguru import logger
from ocsept.data.generation.speed.time_optimal import TimeOptimalStrategy
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.mission import DrivingMission as OcseptDrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics

from trip_stitcher.elevation import ElevationOracle
from trip_stitcher.models import Stop, Trip


@contextmanager
def suppress_output():
    with capture_output():
        yield


# ---------------------------------------------------------------------------
# Hard-coded parameters
# ---------------------------------------------------------------------------
BUS_TYPE = "mega"
AUX_POWER = 2.0e3  # W

# ---------------------------------------------------------------------------
# Hard-coded sample data (route 220, trip 2.TA.96-100-0-j25-1.8.H)
# ---------------------------------------------------------------------------
stop_dict = {
    "8571330": Stop(
        id="8571330", name="Reichenbach i. K., Bahnhof", lon=7.690208, lat=46.625525
    ),
    "8571331": Stop(
        id="8571331", name="Reichenbach i. K., Bären", lon=7.694187, lat=46.625432
    ),
    "8583254": Stop(
        id="8583254", name="Scharnachtal, Halten", lon=7.697223, lat=46.620608
    ),
    "8507766": Stop(
        id="8507766", name="Scharnachtal, Viesen", lon=7.697807, lat=46.617979
    ),
    "8571332": Stop(
        id="8571332", name="Scharnachtal, Schulhaus", lon=7.698158, lat=46.614709
    ),
}

trip = Trip(
    id="2.TA.96-100-0-j25-1.8.H",
    route="96-100-0-j25-1",
    stops=["8571330", "8571331", "8583254", "8507766", "8571332"],
    arrival_times=["18:27:00", "18:28:00", "18:30:00", "18:32:00", "18:33:00"],
)


def main() -> None:
    logger.remove()  # suppress ocsept internal logs

    if BUS_TYPE == "mega":
        from trip_stitcher.vehicles.mega import bus
    elif BUS_TYPE == "maxi":
        from trip_stitcher.vehicles.maxi import bus
    else:
        from trip_stitcher.vehicles.mini import bus

    elevation_oracle = ElevationOracle()
    trip_geometry = trip.download_geometry(stop_dict, elevation_oracle=elevation_oracle)

    itinerary = trip_geometry.create_itinerary()
    comfort = RidingComfort()

    with suppress_output():
        speed_profile = TimeOptimalStrategy().process(itinerary, bus, comfort)

    mission = OcseptDrivingMission(
        name="Fastest possible travel speed",
        time=speed_profile.time,
        speed=speed_profile.speed,
        inclination=itinerary.get_inclination(speed_profile.distance),
        payload="0 kg",
    )
    with suppress_output():
        vehicle_dynamics = LongitudinalVehicleDynamics.of(bus, mission)

    # -------------------------------------------------------------------------
    # 8. Propulsion energy
    # -------------------------------------------------------------------------
    average_velocity = speed_profile.speed[:-1] + speed_profile.speed[1:] / 2
    p_mech = average_velocity * vehicle_dynamics.traction_force[:-1]

    efficiency = 0.9
    propulsion_power = np.where(p_mech >= 0, p_mech / efficiency, p_mech * efficiency)
    propulsion_energy = np.sum(np.diff(speed_profile.time) * propulsion_power).to("J")
    print(f"Propulsion energy: {propulsion_energy}")


if __name__ == "__main__":
    main()
