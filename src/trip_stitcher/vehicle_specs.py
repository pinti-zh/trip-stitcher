from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from ocsept.models.components.vehicles import BatteryBus

from trip_stitcher.vehicles import maxi, mega, mini

VehicleType = Literal["mini", "maxi", "mega"]

_BUILTIN_BUSES = {
    "mini": mini.bus,
    "maxi": maxi.bus,
    "mega": mega.bus,
}


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    base_type: VehicleType
    curb_weight_kg: float
    passenger_payload_kg: float
    aerodynamic_drag_area_m2: float
    rolling_friction_coefficient: float
    battery_capacity_kwh: float
    soc_min: float
    soc_max: float

    def __post_init__(self) -> None:
        _validate_positive("curb_weight_kg", self.curb_weight_kg)
        _validate_nonnegative("passenger_payload_kg", self.passenger_payload_kg)
        _validate_positive("aerodynamic_drag_area_m2", self.aerodynamic_drag_area_m2)
        _validate_positive("rolling_friction_coefficient", self.rolling_friction_coefficient)
        _validate_positive("battery_capacity_kwh", self.battery_capacity_kwh)
        _validate_soc("soc_min", self.soc_min)
        _validate_soc("soc_max", self.soc_max)
        if self.soc_min >= self.soc_max:
            raise ValueError("soc_min must be less than soc_max")
        if self.base_type not in _BUILTIN_BUSES:
            raise ValueError(f"Unknown base vehicle type: {self.base_type}")


def build_battery_bus(vehicle: VehicleSpec) -> BatteryBus:
    bus = deepcopy(_BUILTIN_BUSES[vehicle.base_type])
    fixed_weight_kg = bus.curb_weight.m_as("kg") - bus.chassis.weight.m_as("kg")
    chassis_weight_kg = vehicle.curb_weight_kg - fixed_weight_kg
    if chassis_weight_kg <= 0:
        raise ValueError(
            "curb_weight_kg is too low for the selected base_type's fixed "
            "battery and powertrain weight"
        )

    bus.chassis.weight = f"{chassis_weight_kg} kg"
    bus.chassis.aerodynamic_drag_area = f"{vehicle.aerodynamic_drag_area_m2} m^2"
    bus.chassis.rolling_friction = vehicle.rolling_friction_coefficient
    bus.battery.capacity = f"{vehicle.battery_capacity_kwh} kWh"
    bus.battery.soc_min = vehicle.soc_min
    bus.battery.soc_max = vehicle.soc_max
    return bus


def usable_battery_capacity_j(vehicle: VehicleSpec) -> float:
    return vehicle.battery_capacity_kwh * 3.6e6 * (vehicle.soc_max - vehicle.soc_min)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_soc(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
