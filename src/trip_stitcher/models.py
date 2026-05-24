import uuid
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from ocsept.models.transport.itinerary import Segment, TravelItinerary, Waypoint
from pydantic import BaseModel

from trip_stitcher.elevation import ElevationOracle
from trip_stitcher.utils import limits_from_speed, str_to_datetime, upsample


class Route(BaseModel):
    id: str
    name: str
    trips: list[str]
    vehicle_type: str = "maxi"

    @staticmethod
    def list_from_dataframe(df: pd.DataFrame) -> list["Route"]:
        route_dict = {}
        for route_id, name, trip_id in zip(df["route_id"], df["route_short_name"], df["trip_id"]):
            if route_id not in route_dict:
                route_dict[route_id] = {
                    "name": name,
                    "trips": [trip_id],
                }
            else:
                assert route_dict[route_id]["name"] == name
                if trip_id not in route_dict[route_id]["trips"]:
                    route_dict[route_id]["trips"].append(trip_id)

        routes = []
        for route_id, data_dict in route_dict.items():
            routes.append(Route(id=route_id, name=data_dict["name"], trips=data_dict["trips"]))
        return routes

    def find_representative_trip(self, trip_dict: dict[str, "Trip"]):
        assert len(self.trips) > 0
        trips = [trip_dict[trip_id] for trip_id in self.trips]
        stop_tuples = [tuple(trip.stops) for trip in trips]
        most_common_tuple = Counter(stop_tuples).most_common(1)[0][0]
        for trip in trips:
            if tuple(trip.stops) == most_common_tuple:
                return trip
        # This should never happen
        raise RuntimeError("No representative trip with the most common stops found")


class TripGeometry(BaseModel):
    lon: list[float]
    lat: list[float]
    distance: list[float]
    elevation: list[float]
    is_stop: list[bool]
    speed_limit: list[float]

    @property
    def cumulative_distance(self) -> list[float]:
        cumulative_distance = [0.0]
        for d in self.distance:
            cumulative_distance.append(cumulative_distance[-1] + d)
        return cumulative_distance

    def create_itinerary(self) -> TravelItinerary:
        elements: list[Waypoint | Segment] = [
            Waypoint(maximum_speed_limit="0 km/h", elevation=f"{self.elevation[0]} m")
        ]
        for sl, el, halt, d in zip(
            self.speed_limit, self.elevation[1:], self.is_stop[1:], self.distance
        ):
            if d <= 0:
                continue
            elements.append(Segment(length=f"{d} m", maximum_speed_limit=f"{sl} km/h"))
            waypoint_sl = 0.0 if halt else sl
            elements.append(
                Waypoint(maximum_speed_limit=f"{waypoint_sl} km/h", elevation=f"{el} m")
            )
        return TravelItinerary.from_elements(*elements)


class RouteProfile:
    """Lightweight route profile for energy demand calculation.

    Stores geometry constraints as plain numpy arrays in SI units (metres, m/s,
    radians), replacing the heavier TravelItinerary in performance-critical paths.
    Payload and auxiliary power are not modelled here; callers should treat them
    as zero.
    """

    _MIN_SPEED_M_S = 0.01  # m/s — matches ocsept Segment.minimum_speed_limit default

    def __init__(
        self,
        waypoint_distances: np.ndarray,
        segment_max_speeds: np.ndarray,
        stop_mask: np.ndarray,
        elevations: np.ndarray,
    ) -> None:
        self._waypoint_distances = waypoint_distances  # (M+1,) m
        self._segment_max_speeds = segment_max_speeds  # (M,)  m/s
        self._stop_mask = stop_mask  # (M+1,) bool
        self._elevations = elevations  # (M+1,) m

    @classmethod
    def from_trip_geometry(cls, geom: "TripGeometry") -> "RouteProfile":
        """Build a RouteProfile from TripGeometry, mirroring the d<=0 skip logic
        of TripGeometry.create_itinerary."""
        valid_idx = [i for i, d in enumerate(geom.distance) if d > 0]
        seg_lengths = np.array([geom.distance[i] for i in valid_idx])
        seg_max_speeds = np.array([geom.speed_limit[i] for i in valid_idx]) / 3.6  # km/h → m/s
        elevations = [geom.elevation[0]]
        stop_mask = [True]
        for i in valid_idx:
            elevations.append(geom.elevation[i + 1])
            stop_mask.append(geom.is_stop[i + 1])
        return cls(
            waypoint_distances=np.concatenate([[0.0], np.cumsum(seg_lengths)]),
            segment_max_speeds=seg_max_speeds,
            stop_mask=np.array(stop_mask, dtype=bool),
            elevations=np.array(elevations),
        )

    def _segment_index(self, s: np.ndarray) -> np.ndarray:
        """Index of the segment each sample falls in (previous-fill rule)."""
        idx = np.searchsorted(self._waypoint_distances[:-1], s, side="right") - 1
        return np.clip(idx, 0, len(self._segment_max_speeds) - 1)

    def get_sample_distances(self, step_m: float = 5.0) -> np.ndarray:
        """Sample distances (m) at even intervals merged with waypoint positions."""
        total = self._waypoint_distances[-1]
        evenly_spaced = np.arange(0.0, total, step_m)
        return np.unique(np.concatenate([evenly_spaced, self._waypoint_distances]))

    def get_max_speed_limit(self, s: np.ndarray) -> np.ndarray:
        """Maximum speed limit at each sample distance (m/s)."""
        result = self._segment_max_speeds[self._segment_index(s)].copy()
        stop_dists = self._waypoint_distances[self._stop_mask]
        result[np.isin(s, stop_dists)] = 0.0
        return result

    def get_min_speed_limit(self, s: np.ndarray) -> np.ndarray:
        """Minimum speed limit at each sample distance (m/s)."""
        result = np.full(len(s), self._MIN_SPEED_M_S)
        stop_dists = self._waypoint_distances[self._stop_mask]
        result[np.isin(s, stop_dists)] = 0.0
        return result

    def get_inclination(self, s: np.ndarray) -> np.ndarray:
        """Road inclination at each sample distance (radians)."""
        dz = np.diff(self._elevations)
        dx = np.diff(self._waypoint_distances)
        grade = dz / dx
        return np.arctan(grade[self._segment_index(s)])


class Trip(BaseModel):
    id: str
    route: str
    stops: list[str]
    arrival_times: list[str]
    estimated_energy_demand: float | None = None  # energy demand in joule
    covered_distance: float | None = None  # covered distance in metres

    def download_geometry(
        self,
        stop_dict: dict[str, "Stop"],
        elevation_oracle: ElevationOracle | None = None,
    ) -> TripGeometry:
        coord_string = ";".join(
            f"{round(lon, 6)},{round(lat, 6)}"
            for lat, lon in zip(
                [stop_dict[stop_id].lat for stop_id in self.stops],
                [stop_dict[stop_id].lon for stop_id in self.stops],
            )
        )

        url = f"https://router.project-osrm.org/route/v1/driving/{coord_string}?overview=full&annotations=true&geometries=geojson"

        data = requests.get(url).json()
        if data["code"] != "Ok":
            raise Exception(f"API request failed with code: {data['code']}")

        route_data = data["routes"][0]
        lon = [c[0] for c in route_data["geometry"]["coordinates"]]
        lat = [c[1] for c in route_data["geometry"]["coordinates"]]
        distance = []
        speed = []
        is_stop = [True]
        for leg in route_data["legs"]:
            distance += leg["annotation"]["distance"]
            speed += leg["annotation"]["speed"]
            is_stop += [False] * (len(leg["annotation"]["distance"]) - 1) + [True]

        if elevation_oracle is None:
            elevation_data = [400.0] * len(route_data["geometry"]["coordinates"])
        else:
            sampled_elevation_data = elevation_oracle.get_elevation(
                [value for sample, value in zip(is_stop, lat) if sample],
                [value for sample, value in zip(is_stop, lon) if sample],
            )
            cumulative_distance = [0.0]
            for d in distance:
                cumulative_distance.append(cumulative_distance[-1] + d)
            elevation_data = upsample(cumulative_distance, sampled_elevation_data, is_stop)

        return TripGeometry(
            lon=lon,
            lat=lat,
            distance=distance,
            elevation=elevation_data,
            is_stop=is_stop,
            speed_limit=limits_from_speed([s * 3.6 for s in speed]),
        )

    @staticmethod
    def list_from_dataframe(df: pd.DataFrame) -> list["Trip"]:
        trip_dict = {}
        for trip_id, route_id, stop_id, stop_sequence, arrival_time in zip(
            df["trip_id"],
            df["route_id"],
            df["stop_id"],
            df["stop_sequence"],
            df["arrival_time"],
        ):
            if trip_id not in trip_dict.keys():
                trip_dict[trip_id] = {
                    "route": route_id,
                    "stops": [stop_id],
                    "stop_sequence": [stop_sequence],
                    "arrival_times": [arrival_time],
                }
            else:
                assert route_id == trip_dict[trip_id]["route"]
                trip_dict[trip_id]["stops"].append(stop_id)
                trip_dict[trip_id]["stop_sequence"].append(stop_sequence)
                trip_dict[trip_id]["arrival_times"].append(arrival_time)

        trips = []
        for trip_id, data_dict in trip_dict.items():
            combined_sorted = sorted(
                zip(
                    data_dict["stop_sequence"],
                    data_dict["stops"],
                    data_dict["arrival_times"],
                ),
                key=lambda x: x[0],
            )
            _, sorted_stops, sorted_arrival_times = zip(*combined_sorted)
            trips.append(
                Trip(
                    id=trip_id,
                    route=data_dict["route"],
                    stops=list(sorted_stops),
                    arrival_times=list(sorted_arrival_times),
                )
            )
        return trips


class Stop(BaseModel):
    id: str
    name: str
    lon: float
    lat: float

    @staticmethod
    def create_dummy_depot_from_stops(stops: list["Stop"]) -> "Stop":
        depot_id = uuid.uuid4().hex
        return Stop(
            id=depot_id,
            name=f"Virtual Depot ({depot_id[:4]})",
            lon=sum(stop.lon for stop in stops) / len(stops),
            lat=sum(stop.lat for stop in stops) / len(stops),
        )

    @staticmethod
    def list_from_dataframe(df: pd.DataFrame) -> list["Stop"]:
        stop_dict = {}
        for stop_id, name, lon, lat in zip(
            df["stop_id"], df["stop_name"], df["stop_lon"], df["stop_lat"]
        ):
            if stop_id not in stop_dict.keys():
                stop_dict[stop_id] = {
                    "name": name,
                    "lon": lon,
                    "lat": lat,
                }
            else:
                assert name == stop_dict[stop_id]["name"]
                assert lon == stop_dict[stop_id]["lon"]
                assert lat == stop_dict[stop_id]["lat"]

        stops = []
        for stop_id, data_dict in stop_dict.items():
            stops.append(
                Stop(
                    id=stop_id,
                    name=data_dict["name"],
                    lon=data_dict["lon"],
                    lat=data_dict["lat"],
                )
            )
        return stops


class DrivingMission(BaseModel):
    end_location: str | None = None
    end_time: datetime | None = None
    trips: list[Trip] | None = None

    @property
    def energy_demand(self) -> float:
        ed = 0.0
        assert self.trips is not None
        for trip in self.trips:
            assert trip.estimated_energy_demand is not None
            ed += trip.estimated_energy_demand
        return ed

    def is_addable(self, trip: Trip) -> bool:
        if self.end_location is None:
            return True
        assert self.end_time is not None
        if self.end_location.split(":")[0] == trip.stops[0].split(":")[0]:
            trip_start_time = str_to_datetime(trip.arrival_times[0])
            if trip_start_time >= self.end_time:
                return True
        return False

    def add_trip(self, trip: Trip):
        assert self.is_addable(trip)
        if self.trips is None:
            self.trips = [trip]
        else:
            self.trips.append(trip)
        self.end_location = trip.stops[-1]
        self.end_time = str_to_datetime(trip.arrival_times[-1])

    def add_driving_mission(self, other: "DrivingMission") -> "DrivingMission":
        if other.end_location is None:
            return self
        if self.end_location is None:
            return other
        assert self.is_addable(other.trips[0])
        return DrivingMission(
            end_location=other.end_location,
            end_time=other.end_time,
            trips=self.trips + other.trips,
        )

    def to_input_dict(
        self,
        depot_ids: list[str] | None = None,
        battery_capacity: float = 300.0 * 3.6e6,
        max_charging_power: float = 150000.0,
    ) -> dict:
        num_vehicles = 1
        time = []
        energy_demand = []
        depot_charge = []
        in_depot = True
        for trip in self.trips:
            trip_start_time = int(
                (
                    str_to_datetime(trip.arrival_times[0]) - str_to_datetime("00:00:00")
                ).total_seconds()
            )
            trip_stop_time = int(
                (
                    str_to_datetime(trip.arrival_times[-1]) - str_to_datetime("00:00:00")
                ).total_seconds()
            )
            assert trip_start_time < trip_stop_time
            if len(time) == 0 or time[-1] < trip_start_time:
                energy_demand.append(0.0)
                time.append(trip_start_time)
                depot_charge.append(in_depot)
            energy_demand.append(trip.estimated_energy_demand or 0.0)
            time.append(trip_stop_time)
            depot_charge.append(False)
            in_depot = trip.stops[-1] in depot_ids
        assert all(t1 < t2 for t1, t2 in zip(time[:-1], time[1:]))
        time, energy_demand, depot_charge = fit_to_24hs(time, energy_demand, depot_charge)
        return {
            "num_vehicles": num_vehicles,
            "time": time,
            "energy_demand": [energy_demand],
            "depot_charge": [depot_charge],
            "max_charging_power": max_charging_power,
            "battery_capacity": [battery_capacity],
            "is_battery": [False],
        }


def fit_to_24hs(
    time: list[int], energy_demand: list[float], depot_charge: list[bool]
) -> tuple[list[int], list[float], list[bool]]:
    """
    Disclaimer: This code is kinda ugly, might want to refactor in the future.
    """
    seconds_in_a_day = 60 * 60 * 24

    adjusted_time, adjusted_energy_demand, adjusted_depot_charge = [], [], []

    # case all good
    if time[-1] == seconds_in_a_day:
        return time, energy_demand, depot_charge

    # case more than one day and the day mark is hit exactly
    if seconds_in_a_day in time:
        for i, t in enumerate(time):
            if t == seconds_in_a_day:
                adjusted_time = [v % seconds_in_a_day for v in time[i + 1 :]] + time[: i + 1]
                adjusted_energy_demand = energy_demand[i + 1 :] + energy_demand[: i + 1]
                adjusted_depot_charge = depot_charge[i + 1 :] + depot_charge[: i + 1]
                return adjusted_time, adjusted_energy_demand, adjusted_depot_charge

    # case one interval needs to split
    if not all(t <= seconds_in_a_day for t in time):
        for i, t in enumerate(time):
            if t > seconds_in_a_day:
                t_1 = seconds_in_a_day - time[i - 1]
                t_2 = t - seconds_in_a_day
                ed_1 = energy_demand[i] * t_1 / (t_1 + t_2)
                ed_2 = energy_demand[i] * t_1 / (t_1 + t_2)
                adjusted_time = (
                    [v % seconds_in_a_day for v in time[i:]] + time[:i] + [seconds_in_a_day]
                )
                adjusted_energy_demand = (
                    [ed_2] + energy_demand[i + 1 :] + energy_demand[:i] + [ed_1]
                )
                adjusted_depot_charge = depot_charge[i:] + depot_charge[:i] + [depot_charge[i]]
                return adjusted_time, adjusted_energy_demand, adjusted_depot_charge

    # case day needs completion
    return time + [seconds_in_a_day], energy_demand + [0.0], depot_charge + [True]


class RadiusQuery(BaseModel):
    radius: float
    lat: float
    lon: float
