import uuid
from collections import Counter
from datetime import datetime

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
        for sl, el, halt, d in zip(self.speed_limit, self.elevation[1:], self.is_stop[1:], self.distance):
            if d <= 0:
                continue
            elements.append(Segment(length=f"{d} m", maximum_speed_limit=f"{sl} km/h"))
            waypoint_sl = 0.0 if halt else sl
            elements.append(Waypoint(maximum_speed_limit=f"{waypoint_sl} km/h", elevation=f"{el} m"))
        return TravelItinerary.from_elements(*elements)


class Trip(BaseModel):
    id: str
    route: str
    stops: list[str]
    arrival_times: list[str]
    estimated_energy_demand: float | None = None  # energy demand in joule

    def download_geometry(
        self, stop_dict: dict[str, "Stop"], elevation_oracle: ElevationOracle | None = None
    ) -> TripGeometry:
        coord_string = ";".join(
            f"{round(lon, 6)},{round(lat, 6)}"
            for lat, lon in zip(
                [stop_dict[stop_id].lat for stop_id in self.stops], [stop_dict[stop_id].lon for stop_id in self.stops]
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
                zip(data_dict["stop_sequence"], data_dict["stops"], data_dict["arrival_times"]), key=lambda x: x[0]
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
            name=f"Depot-{depot_id[:4]}",
            lon=sum(stop.lon for stop in stops) / len(stops),
            lat=sum(stop.lat for stop in stops) / len(stops),
        )

    @staticmethod
    def list_from_dataframe(df: pd.DataFrame) -> list["Stop"]:
        stop_dict = {}
        for stop_id, name, lon, lat in zip(df["stop_id"], df["stop_name"], df["stop_lon"], df["stop_lat"]):
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
            stops.append(Stop(id=stop_id, name=data_dict["name"], lon=data_dict["lon"], lat=data_dict["lat"]))
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


class RadiusQuery(BaseModel):
    radius: float
    lat: float
    lon: float
