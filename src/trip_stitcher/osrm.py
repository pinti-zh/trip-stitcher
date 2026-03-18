import requests
from loguru import logger

logger.remove()

from ocsept.models.transport.itinerary import Segment, TravelItinerary, Waypoint

from trip_stitcher.elevation import Oracle
from trip_stitcher.models import Stop, Trip

elevation_oracle = Oracle()


def snap_to_tens(value: float) -> int:
    # This is a terrible but correct implementation
    assert value >= 0
    result = 0
    while result < value + 3:
        result += 10
    return result


def add_routing_to_trip(trip: Trip, stop_dict: dict[str, Stop]) -> Trip:
    coord_string = ";".join(
        f"{round(lon, 6)},{round(lat, 6)}"
        for lat, lon in zip(
            [stop_dict[stop_id].lat for stop_id in trip.stops], [stop_dict[stop_id].lon for stop_id in trip.stops]
        )
    )
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_string}?overview=full&annotations=true&geometries=geojson"

    data = requests.get(url).json()
    if data["code"] == "Ok":
        route_lon = [c[0] for c in data["routes"][0]["geometry"]["coordinates"]]
        route_lat = [c[1] for c in data["routes"][0]["geometry"]["coordinates"]]
        return Trip(
            id=trip.id,
            route=trip.route,
            route_lat=route_lat,
            route_lon=route_lon,
            stops=trip.stops,
            arrival_times=trip.arrival_times,
            estimated_energy_demand=trip.estimated_energy_demand,
        )
    else:
        raise RuntimeError(f"API request failed with code {data['code']}")


def create_itinerary_from_trip(trip: Trip, stop_dict: dict[str, Stop]) -> TravelItinerary:
    dwell_time = 10
    elevation = elevation_oracle.get_elevation(
        [stop_dict[stop_id].lat for stop_id in trip.stops], [stop_dict[stop_id].lon for stop_id in trip.stops]
    )

    coord_string = ";".join(
        f"{round(lon, 6)},{round(lat, 6)}"
        for lat, lon in zip(
            [stop_dict[stop_id].lat for stop_id in trip.stops], [stop_dict[stop_id].lon for stop_id in trip.stops]
        )
    )
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_string}?overview=simplified&annotations=true"

    data = requests.get(url).json()
    if data["code"] == "Ok":
        elements: list[Waypoint | Segment] = [Waypoint(maximum_speed_limit="0 km/h", elevation=f"{elevation[0]} m")]
        route = data["routes"][0]["legs"]
        total_length = 0
        assert len(route) == len(elevation) - 1
        for i, leg in enumerate(route):
            annotation = leg["annotation"]
            partial_distance_sum = [sum(annotation["distance"][:i]) for i in range(len(annotation["distance"]) + 1)]
            start_elevation = elevation[i]
            end_elevation = elevation[i + 1]
            for leg_distance, length, speed_limit in zip(
                partial_distance_sum, annotation["distance"], annotation["speed"]
            ):
                total_length += length
                if length > 5.0e-1:
                    speed_limit = snap_to_tens(speed_limit * 3.6) / 3.6
                    elements.append(Segment(length=f"{length} m", maximum_speed_limit=f"{speed_limit} m/s"))
                    alpha = leg_distance / leg["distance"]
                    interpolated_elevation = (1 - alpha) * start_elevation + alpha * end_elevation
                    elements.append(
                        Waypoint(
                            minimum_speed_limit="0 m/s",
                            maximum_speed_limit=f"{speed_limit} m/s",
                            elevation=f"{interpolated_elevation} m",
                        )
                    )
            elements = elements[:-1] + [
                Waypoint(
                    minimum_speed_limit="0 m/s",
                    maximum_speed_limit="0 m/s",
                    elevation=f"{end_elevation} m",
                    dwell_time=f"{dwell_time} s",
                )
            ]
        logger.debug(f"Number of elements: {len(elements)}")
        return TravelItinerary.from_elements(*elements)
    else:
        raise RuntimeError(f"API request failed with code {data['code']}")
