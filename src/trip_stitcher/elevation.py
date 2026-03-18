import time

import requests_cache


class Oracle:
    def __init__(self):
        self._url = "https://api.opentopodata.org/v1/eudem25m"
        self._max_locations_per_request = 100
        self._last_request_ts = 0
        self._request_session = requests_cache.CachedSession(cache_name="api_cache", expire_after=3600 * 24 * 30)

    def get_elevation(self, latitude: list[float], longitude: list[float]) -> list[float]:
        assert len(latitude) == len(longitude)
        elevation = []
        while len(latitude) > 0 and len(longitude) > 0:
            lat_chunk = latitude[:self._max_locations_per_request]
            lon_chunk = longitude[:self._max_locations_per_request]
            location_string = "?locations=" + "|".join(f"{lat},{lon}" for lat, lon in zip(lat_chunk, lon_chunk))
            while time.time() < self._last_request_ts + 1.0:
                pass # wait one second because of rate limit
            response = self._request_session.get(self._url + location_string)
            data = response.json()
            if data["status"] != "OK":
                raise RuntimeError(f"Elevation request failed with status {data['status']}")
            if not response.from_cache:
                self._last_request_ts = time.time()
            elevation += [item["elevation"] for item in data["results"]]
            latitude = latitude[self._max_locations_per_request:]
            longitude = longitude[self._max_locations_per_request:]
        return elevation