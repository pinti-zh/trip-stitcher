import uuid
from datetime import timedelta
from typing import Callable

from trip_stitcher.models import DrivingMission, Stop, Trip
from trip_stitcher.utils import datetime_to_str, str_to_datetime


def driving_mission_ends_at_trip_start(driving_mission: DrivingMission, trip: Trip) -> bool:
    if driving_mission.end_location is None:
        return True
    return driving_mission.end_location == trip.stops[0]


def driving_mission_ends_before_trip(driving_mission: DrivingMission, trip: Trip) -> bool:
    if driving_mission.end_time is None:
        return True
    return str_to_datetime(trip.arrival_times[0]) > driving_mission.end_time


def trip_exceeds_energy_capacity(driving_mission: DrivingMission, trip: Trip, max_capacity=300 * 3.6e6) -> bool:
    if trip.estimated_energy_demand is None:
        return False
    if driving_mission.end_time is None:
        return trip.estimated_energy_demand > max_capacity
    return trip.estimated_energy_demand + driving_mission.energy_demand > max_capacity


def stitch_trips_into_driving_missions(
    trips: list[Trip], is_addable: Callable[[DrivingMission, Trip], bool], lifo: bool = False
) -> list[DrivingMission]:
    trips = sorted(trips, key=lambda t: t.arrival_times[0])
    driving_missions = []
    for trip in trips:
        if lifo:
            driving_missions = sorted(driving_missions, key=lambda m: m.end_time)[::-1]
        else:
            driving_missions = sorted(driving_missions, key=lambda m: m.end_time)
        for driving_mission in filter(lambda dm: is_addable(dm, trip), driving_missions):
            driving_mission.add_trip(trip)
            break
        else:
            new_driving_mission = DrivingMission()
            new_driving_mission.add_trip(trip)
            driving_missions.append(new_driving_mission)
    return sorted(driving_missions, key=lambda dm: dm.trips[0].arrival_times[0])


def add_depot_trips(
    driving_missions: list[DrivingMission],
    depot: Stop,
    time_function: Callable[[str, str], timedelta] | None = None,
    energy_demand_function: Callable[[str, str], float] | None = None,
) -> list[DrivingMission]:
    for driving_mission in driving_missions:
        assert driving_mission.trips is not None and len(driving_mission.trips) > 0

        first_stop = driving_mission.trips[0].stops[0]
        last_stop = driving_mission.trips[-1].stops[-1]

        if time_function is not None:
            time_to_first_stop = time_function(depot.id, first_stop)
            time_from_last_stop = time_function(depot.id, last_stop)
        else:
            time_to_first_stop = timedelta(seconds=0)
            time_from_last_stop = timedelta(seconds=0)

        if energy_demand_function is not None:
            energy_to_first_stop = energy_demand_function(depot.id, first_stop)
            energy_from_last_stop = energy_demand_function(depot.id, last_stop)
        else:
            energy_to_first_stop = 0.0
            energy_from_last_stop = 0.0

        departure_datetime = str_to_datetime(driving_mission.trips[0].arrival_times[0]) - time_to_first_stop
        departure_time = datetime_to_str(departure_datetime, is_next_day=departure_datetime.day == 2)
        arrival_datetime = str_to_datetime(driving_mission.trips[-1].arrival_times[-1]) + time_from_last_stop
        arrival_time = datetime_to_str(arrival_datetime, is_next_day=arrival_datetime.day == 2)

        trip_from_depot = Trip(
            id=uuid.uuid4().hex,
            route=driving_mission.trips[0].route,
            stops=[depot.id, first_stop],
            arrival_times=[departure_time, driving_mission.trips[0].arrival_times[0]],
            estimated_energy_demand=energy_to_first_stop,
        )

        trip_to_depot = Trip(
            id=uuid.uuid4().hex,
            route=driving_mission.trips[0].route,
            stops=[last_stop, depot.id],
            arrival_times=[driving_mission.trips[-1].arrival_times[-1], arrival_time],
            estimated_energy_demand=energy_from_last_stop,
        )
        driving_mission.trips = [trip_from_depot] + driving_mission.trips
        driving_mission.trips.append(trip_to_depot)
    return driving_missions
