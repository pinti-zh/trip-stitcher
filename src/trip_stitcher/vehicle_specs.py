from __future__ import annotations

from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Literal, Mapping

from ocsept.models.components.vehicles import BatteryBus

from trip_stitcher.vehicles import maxi, mega, mini

VehicleType = Literal["mini", "maxi", "mega"]

_VEHICLE_MODULES = {
    "mini": mini,
    "maxi": maxi,
    "mega": mega,
}


@dataclass(frozen=True, slots=True)
class VehicleSpecOverride:
    curb_weight_kg: float | None = None
    passenger_payload_kg: float | None = None
    aerodynamic_drag_area_m2: float | None = None
    rolling_friction_coefficient: float | None = None
    battery_capacity_j: float | None = None
    battery_capacity_kwh: float | None = None
    soc_min: float | None = None
    soc_max: float | None = None

    def __post_init__(self) -> None:
        _validate_positive("curb_weight_kg", self.curb_weight_kg)
        _validate_positive("passenger_payload_kg", self.passenger_payload_kg)
        _validate_positive("aerodynamic_drag_area_m2", self.aerodynamic_drag_area_m2)
        _validate_positive("rolling_friction_coefficient", self.rolling_friction_coefficient)
        _validate_positive("battery_capacity_j", self.battery_capacity_j)
        _validate_positive("battery_capacity_kwh", self.battery_capacity_kwh)

        if self.battery_capacity_j is not None and self.battery_capacity_kwh is not None:
            raise ValueError("only one of battery_capacity_j or battery_capacity_kwh may be set")

        _validate_soc("soc_min", self.soc_min)
        _validate_soc("soc_max", self.soc_max)
        if self.soc_min is not None and self.soc_max is not None and self.soc_min >= self.soc_max:
            raise ValueError("soc_min must be less than soc_max")

    def cache_key(self) -> tuple[float | None, ...]:
        return astuple(self)


def get_builtin_bus(bus_type: VehicleType) -> BatteryBus:
    try:
        return _VEHICLE_MODULES[bus_type].bus
    except KeyError as exc:
        raise ValueError(f"Unknown bus type: {bus_type}") from exc


def build_battery_bus(
    bus_type: VehicleType, overrides: VehicleSpecOverride | None = None
) -> BatteryBus:
    bus = deepcopy(get_builtin_bus(bus_type))
    if overrides is None:
        return bus

    if overrides.curb_weight_kg is not None:
        bus.chassis.weight = f"{overrides.curb_weight_kg} kg"
    if overrides.aerodynamic_drag_area_m2 is not None:
        bus.chassis.aerodynamic_drag_area = f"{overrides.aerodynamic_drag_area_m2} m^2"
    if overrides.rolling_friction_coefficient is not None:
        bus.chassis.rolling_friction = overrides.rolling_friction_coefficient
    if overrides.battery_capacity_j is not None:
        bus.battery.capacity = f"{overrides.battery_capacity_j} J"
    if overrides.battery_capacity_kwh is not None:
        bus.battery.capacity = f"{overrides.battery_capacity_kwh} kWh"
    if overrides.soc_min is not None:
        bus.battery.soc_min = overrides.soc_min
    if overrides.soc_max is not None:
        bus.battery.soc_max = overrides.soc_max

    return bus


def usable_battery_capacity_j(
    bus_type: VehicleType, overrides: VehicleSpecOverride | None = None
) -> float:
    bus = build_battery_bus(bus_type, overrides)
    return bus.battery.capacity.m_as("J") * (bus.battery.soc_max - bus.battery.soc_min)


def build_vehicle_capacity_map(
    overrides: Mapping[VehicleType, VehicleSpecOverride] | None = None,
) -> dict[str, float]:
    overrides = overrides or {}
    return {
        bus_type: usable_battery_capacity_j(bus_type, overrides.get(bus_type))
        for bus_type in _VEHICLE_MODULES
    }


def _validate_positive(name: str, value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_soc(name: str, value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
