import uuid
from datetime import timedelta
from functools import partial
from typing import Callable

from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching

from trip_stitcher.models import DrivingMission, Route, Stop, Trip
from trip_stitcher.utils import datetime_to_str, str_to_datetime
from trip_stitcher.vehicles import maxi, mega, mini

DEFAULT_VEHICLE_CAPACITY_MAP: dict[str, float] = {
    "maxi": maxi.bus.battery.capacity.m_as("J")
    * (maxi.bus.battery.soc_max - maxi.bus.battery.soc_min),
    "mega": mega.bus.battery.capacity.m_as("J")
    * (mega.bus.battery.soc_max - mega.bus.battery.soc_min),
    "mini": mini.bus.battery.capacity.m_as("J")
    * (mini.bus.battery.soc_max - mini.bus.battery.soc_min),
}


def driving_mission_and_trip_match_vehicles(
    driving_mission: DrivingMission, trip: Trip, route_dict: dict[str, Route] | None = None
) -> bool:
    if route_dict is None:
        raise ValueError("route_dict must not be None")
    return (
        route_dict[trip.route].vehicle_type
        == route_dict[driving_mission.trips[-1].route].vehicle_type
    )


def driving_missions_match_vehicles(
    first_dm: DrivingMission, second_dm: DrivingMission, route_dict: dict[str, Route] | None = None
) -> bool:
    if route_dict is None:
        raise ValueError("route_dict must not be None")
    return (
        route_dict[first_dm.trips[-1].route].vehicle_type
        == route_dict[second_dm.trips[0].route].vehicle_type
    )


def driving_mission_ends_at_trip_start(driving_mission: DrivingMission, trip: Trip) -> bool:
    if driving_mission.end_location is None:
        return True
    return driving_mission.end_location == trip.stops[0]


def driving_mission_ends_before_trip(driving_mission: DrivingMission, trip: Trip) -> bool:
    if driving_mission.end_time is None:
        return True
    return str_to_datetime(trip.arrival_times[0]) >= driving_mission.end_time


def trip_within_vehicle_energy_capacity(
    driving_mission: DrivingMission,
    trip: Trip,
    route_dict: dict[str, Route],
    vehicle_capacity_map: dict[str, float],
) -> bool:
    if trip.estimated_energy_demand is None:
        raise ValueError(f"trip '{trip.id}' has no estimated energy demand")
    vehicle_type = route_dict[trip.route].vehicle_type
    max_capacity = vehicle_capacity_map.get(vehicle_type)
    if max_capacity is None:
        raise ValueError(f"no capacity defined for vehicle type '{vehicle_type}'")
    if driving_mission.end_time is None:
        return trip.estimated_energy_demand <= max_capacity
    return trip.estimated_energy_demand + driving_mission.energy_demand <= max_capacity


def build_default_is_addable(
    route_dict: dict[str, Route] | None = None,
    vehicle_capacity_map: dict[str, float] | None = None,
) -> Callable[[DrivingMission, Trip], bool]:
    """
    Build a composite predicate to determine whether a trip can be added
    to a driving mission.

    The returned function evaluates a set of conditions that must all be met
    for a (DrivingMission, Trip) pair to be considered compatible. Additional
    constraints are included depending on the provided arguments.

    Args:
        route_dict: Optional mapping used to enforce vehicle compatibility
            between driving missions and trips, and to look up vehicle type
            for energy capacity checks.
        vehicle_capacity_map: Mapping from vehicle type string to usable energy
            capacity in joules. Defaults to DEFAULT_VEHICLE_CAPACITY_MAP when
            route_dict is provided. Supply a custom map to support additional
            vehicle types beyond the built-in maxi/mega/mini.

    Returns:
        A callable that takes a DrivingMission and a Trip and returns True
        if all configured conditions are satisfied, otherwise False.
    """

    predicates = [
        driving_mission_ends_before_trip,
        driving_mission_ends_at_trip_start,
    ]

    if route_dict is not None:
        cap_map = (
            vehicle_capacity_map
            if vehicle_capacity_map is not None
            else DEFAULT_VEHICLE_CAPACITY_MAP
        )
        predicates.append(partial(driving_mission_and_trip_match_vehicles, route_dict=route_dict))
        predicates.append(
            partial(
                trip_within_vehicle_energy_capacity,
                route_dict=route_dict,
                vehicle_capacity_map=cap_map,
            )
        )

    def is_addable(dm: DrivingMission, t: Trip) -> bool:
        return all(p(dm, t) for p in predicates)

    return is_addable


def build_default_is_stitchable(
    minimum_timedelta: timedelta = timedelta(hours=4),
    route_dict: dict[str, Route] | None = None,
) -> Callable[[DrivingMission, DrivingMission], bool]:
    """
    Build a composite predicate to determine whether two driving missions
    can be stitched together.

    The returned function evaluates a set of conditions that must all be met
    for a (DrivingMission, DrivingMission) pair to be considered compatible.
    Additional constraints are included depending on the provided arguments.

    Args:
        minimum_timedelta: Minimum time gap that must exist between the end of the
            first driving mission and the start of the second for them to be stitchable.
        route_dict: Optional mapping used to enforce vehicle compatibility
            between driving missions.

    Returns:
        A callable that takes two DrivingMissions and returns True if all
        configured conditions are satisfied, otherwise False.
    """

    def timedelta_check(first_dm: DrivingMission, second_dm: DrivingMission) -> bool:
        assert first_dm.end_time is not None
        assert len(second_dm.trips) > 0 and second_dm.trips[0].arrival_times is not None
        dt = str_to_datetime(second_dm.trips[0].arrival_times[0]) - first_dm.end_time
        return dt >= minimum_timedelta

    predicates: list[Callable[[DrivingMission, DrivingMission], bool]] = [timedelta_check]

    if route_dict is not None:
        predicates.append(partial(driving_missions_match_vehicles, route_dict=route_dict))

    def is_stitchable(first_dm: DrivingMission, second_dm: DrivingMission) -> bool:
        return all(p(first_dm, second_dm) for p in predicates)

    return is_stitchable


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


def stitch_driving_missions(
    driving_missions: list[DrivingMission],
    is_stitchable: Callable[[DrivingMission, DrivingMission], bool],
) -> list[DrivingMission]:
    dm_lookup = {}
    nodes = []
    for i, first_dm in enumerate(driving_missions):
        node = []
        dm_lookup[i] = first_dm
        for second_dm in driving_missions:
            if is_stitchable(first_dm, second_dm):
                node.append(1)
            else:
                node.append(0)
        nodes.append(node)
    graph = csr_array(nodes)
    matching = maximum_bipartite_matching(graph, perm_type="column")
    index_group = list(range(len(driving_missions)))
    for i, j in enumerate(matching):
        if j == -1:
            continue
        u = index_group[i]
        v = index_group[j]
        dm_lookup[u] = dm_lookup[u].add_driving_mission(dm_lookup[v])
        index_group[j] = u

    merged_driving_missions = []
    for index in set(index_group):
        merged_driving_missions.append(dm_lookup[index])
    return merged_driving_missions


def add_depot_trips(
    driving_missions: list[DrivingMission],
    depot: Stop,
    time_function: Callable[[str, str], timedelta]
    | None = None,  # function that maps depot ids to timedelta
    energy_demand_function: Callable[[str, str], float]
    | None = None,  # function that maps depot ids to energy demand
    covered_distance_function: Callable[[str, str], float]
    | None = None,  # function that maps depot ids to covered distance in metres
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

        if covered_distance_function is not None:
            distance_to_first_stop = covered_distance_function(depot.id, first_stop)
            distance_from_last_stop = covered_distance_function(depot.id, last_stop)
        else:
            distance_to_first_stop = 0.0
            distance_from_last_stop = 0.0

        departure_datetime = (
            str_to_datetime(driving_mission.trips[0].arrival_times[0]) - time_to_first_stop
        )
        departure_time = datetime_to_str(
            departure_datetime, is_next_day=departure_datetime.day == 2
        )
        arrival_datetime = (
            str_to_datetime(driving_mission.trips[-1].arrival_times[-1]) + time_from_last_stop
        )
        arrival_time = datetime_to_str(arrival_datetime, is_next_day=arrival_datetime.day == 2)

        trip_from_depot = Trip(
            id=f"virtual-depot-trip-{uuid.uuid4().hex[:4]}",
            route=driving_mission.trips[0].route,
            stops=[depot.id, first_stop],
            arrival_times=[departure_time, driving_mission.trips[0].arrival_times[0]],
            estimated_energy_demand=energy_to_first_stop,
            covered_distance=distance_to_first_stop,
        )

        trip_to_depot = Trip(
            id=f"virtual-depot-trip-{uuid.uuid4().hex[:4]}",
            route=driving_mission.trips[0].route,
            stops=[last_stop, depot.id],
            arrival_times=[driving_mission.trips[-1].arrival_times[-1], arrival_time],
            estimated_energy_demand=energy_from_last_stop,
            covered_distance=distance_from_last_stop,
        )
        driving_mission.trips = [trip_from_depot] + driving_mission.trips
        driving_mission.add_trip(trip_to_depot)
    return driving_missions
