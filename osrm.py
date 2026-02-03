import requests
from loguru import logger

logger.remove()

from ocsept.models.transport.itinerary import Segment, TravelItinerary, Waypoint

from elevation import Oracle
from models import Trip
from pipeline_utils import get_default_parser, run_pipeline, setup_logger

elevation_oracle = Oracle()


def main(trip: Trip) -> TravelItinerary:
    dwell_time = 10
    coord_string = ";".join(f"{round(lon, 6)},{round(lat, 6)}" for lat, lon in zip(trip.lat, trip.lon))
    elevation = elevation_oracle.get_elevation(trip.lat, trip.lon)

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
                    speed_limit = int(speed_limit + 1)
                    elements.append(Segment(length=f"{length} m", maximum_speed_limit=f"{speed_limit} m/s"))
                    alpha = leg_distance / leg["distance"]
                    interpolated_elevation = (1 - alpha) * start_elevation + alpha * end_elevation
                    elements.append(
                        Waypoint(minimum_speed_limit="0 m/s", maximum_speed_limit=f"{speed_limit} m/s", elevation=f"{interpolated_elevation} m")
                    )
            elements = elements[:-1] + [
                Waypoint(minimum_speed_limit="0 m/s", maximum_speed_limit="0 m/s", elevation=f"{end_elevation} m", dwell_time=f"{dwell_time} s")
            ]
        # elements = elements[:583]
        logger.debug(f"Number of elements: {len(elements)}")
        return TravelItinerary.from_elements(*elements)
    else:
        raise RuntimeError(f"API request failed with code {data['code']}")


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    setup_logger(args.debug)
    run_pipeline(main, Trip, args)
