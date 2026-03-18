import numpy as np
from sklearn.neighbors import BallTree

from trip_stitcher.models import RadiusQuery, Route, Stop, Trip


class RouteLocator:
    def __init__(self, routes: list[Route], trips: list[Trip], stops: list[Stop]):
        self.routes = routes
        self.trips = trips
        self.stops = stops

        # dictionaries for better lookup
        self._route_dict = dict((route.id, route) for route in self.routes)
        self._trip_dict = dict((trip.id, trip) for trip in self.trips)
        self._stop_to_routes_map = self._get_stop_to_routes_map()

        # efficient lookup data structure
        self._stop_coordinates = np.array([[np.radians(stop.lat), np.radians(stop.lon)] for stop in stops])
        self._ball_tree = BallTree(self._stop_coordinates, metric="haversine")

    def _get_stop_to_routes_map(self) -> dict[str, str]:
        stop_to_routes_map = {}
        for route in self._route_dict.values():
            for trip_id in route.trips:
                for stop_id in self._trip_dict[trip_id].stops:
                    if stop_id not in stop_to_routes_map:
                        stop_to_routes_map[stop_id] = [route.id]
                    elif route.id not in stop_to_routes_map[stop_id]:
                        stop_to_routes_map[stop_id].append(route.id)
        return stop_to_routes_map

    def radius_query(self, query: RadiusQuery) -> list[Route]:
        """
        Return all routes that have at least one stop within the given radius of the specified coordinates.

        The query is performed using a BallTree with haversine distance. The input
        coordinates are interpreted as latitude and longitude in degrees, and the
        radius is given in meters.

        Args:
            query (RadiusQuery):
                A RadiusQuery object containing radius, latitude and longitude

        Returns:
            list[Route]:
                A list of unique Route objects that have at least one stop located
                at most ``radius`` meters from the given coordinates.

        Notes:
            - Internally converts coordinates to radians for haversine distance.
            - Distance is computed assuming an Earth radius of 6'371'000 meters.
        """
        earth_radius_m = 6_371_000
        stop_indices = self._ball_tree.query_radius(
            [list(map(np.radians, [query.lat, query.lon]))],
            r=query.radius / earth_radius_m,
        )[0]

        routes = set()
        for index in stop_indices:
            routes.update(self._stop_to_routes_map[self.stops[index].id])

        return [self._route_dict[route_id] for route_id in routes]
