from pydantic import BaseModel


class Trip(BaseModel):
    name: str
    route: str
    stops: list[str]
    stop_sequence: list[int]
    arrival_times: list[str]
    lon: list[float]
    lat: list[float]

    def truncate(self, num: int) -> "Trip":
        return Trip(
            name=self.name,
            route=self.route,
            stops=self.stops[:num],
            stop_sequence=self.stop_sequence[:num],
            arrival_times=self.arrival_times[:num],
            lon=self.lon[:num],
            lat=self.lat[:num],
        )


class EnergyDemand(BaseModel):
    magnitude: float
    units: str
    time: float
