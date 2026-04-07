from typing import Callable

from trip_stitcher.models import DrivingMission, Trip
from trip_stitcher.utils import str_to_datetime


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
