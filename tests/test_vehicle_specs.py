import numpy as np
import pytest
from ocsept.data.generation.speed import SpeedProfile
from optool.uom import Quantity

from trip_stitcher.energy_demand_estimator import EnergyDemandEstimator, _CasadiStrategyTag
from trip_stitcher.models import DrivingMission, Route, Trip, TripGeometry
from trip_stitcher.stitching import build_default_is_addable
from trip_stitcher.vehicle_specs import VehicleSpec, build_battery_bus, usable_battery_capacity_j
from trip_stitcher.vehicles import maxi


def _maxi_spec(**updates) -> VehicleSpec:
    values = {
        "base_type": "maxi",
        "curb_weight_kg": 18000.0,
        "passenger_payload_kg": 0.0,
        "aerodynamic_drag_area_m2": 5.0,
        "rolling_friction_coefficient": 0.012,
        "battery_capacity_kwh": 556.0,
        "soc_min": 0.2,
        "soc_max": 0.8,
    }
    values.update(updates)
    return VehicleSpec(**values)


def test_vehicle_spec_validation():
    with pytest.raises(ValueError, match="curb_weight_kg"):
        _maxi_spec(curb_weight_kg=0.0)
    with pytest.raises(ValueError, match="passenger_payload_kg"):
        _maxi_spec(passenger_payload_kg=-1.0)
    with pytest.raises(ValueError, match="soc_min"):
        _maxi_spec(soc_min=-0.1)
    with pytest.raises(ValueError, match="soc_min must be less"):
        _maxi_spec(soc_min=0.8, soc_max=0.2)
    with pytest.raises(ValueError, match="Unknown base"):
        _maxi_spec(base_type="bogus")


def test_build_battery_bus_applies_public_fields_and_preserves_template_fields():
    vehicle = _maxi_spec(
        curb_weight_kg=19000.0,
        passenger_payload_kg=500.0,
        aerodynamic_drag_area_m2=4.4,
        rolling_friction_coefficient=0.02,
        battery_capacity_kwh=400.0,
        soc_min=0.1,
        soc_max=0.9,
    )

    bus = build_battery_bus(vehicle)

    assert bus is not maxi.bus
    assert bus.curb_weight.m_as("kg") == pytest.approx(19000.0)
    assert bus.chassis.aerodynamic_drag_area.m_as("m**2") == pytest.approx(4.4)
    assert bus.chassis.rolling_friction == pytest.approx(0.02)
    assert bus.battery.capacity.m_as("kWh") == pytest.approx(400.0)
    assert bus.battery.soc_min == pytest.approx(0.1)
    assert bus.battery.soc_max == pytest.approx(0.9)

    assert bus.powertrain.final_drive_efficiency == maxi.bus.powertrain.final_drive_efficiency
    assert bus.powertrain.final_drive_transmission_ratio == (
        maxi.bus.powertrain.final_drive_transmission_ratio
    )
    assert bus.chassis.wheel_radius == maxi.bus.chassis.wheel_radius
    assert bus.chassis.wheel_inertia == maxi.bus.chassis.wheel_inertia
    assert bus.battery.chemistry == maxi.bus.battery.chemistry
    assert bus.battery.c_rate_limit_continuous_charging == (
        maxi.bus.battery.c_rate_limit_continuous_charging
    )
    assert bus.battery.c_rate_limit_continuous_discharging == (
        maxi.bus.battery.c_rate_limit_continuous_discharging
    )
    assert bus.battery.c_rate_limit_peak_charging == maxi.bus.battery.c_rate_limit_peak_charging
    assert bus.battery.c_rate_limit_peak_discharging == (
        maxi.bus.battery.c_rate_limit_peak_discharging
    )
    assert bus.powertrain.motors[0][1].power_limit == maxi.bus.powertrain.motors[0][1].power_limit
    assert bus.powertrain.motors[0][1].torque_limit == (
        maxi.bus.powertrain.motors[0][1].torque_limit
    )


def test_build_battery_bus_rejects_unachievable_curb_weight():
    with pytest.raises(ValueError, match="curb_weight_kg is too low"):
        build_battery_bus(_maxi_spec(curb_weight_kg=100.0))


def test_usable_battery_capacity_uses_caller_spec():
    vehicle = _maxi_spec(battery_capacity_kwh=2.0, soc_min=0.25, soc_max=0.75)

    assert usable_battery_capacity_j(vehicle) == pytest.approx(3600000.0)


def test_stitching_requires_explicit_capacity_map_with_route_dict():
    route_dict = {"r1": Route(id="r1", name="Route 1", trips=["t1"], vehicle_type="maxi")}

    with pytest.raises(ValueError, match="vehicle_capacity_map"):
        build_default_is_addable(route_dict=route_dict)


def test_stitching_capacity_uses_caller_map():
    route_dict = {"r1": Route(id="r1", name="Route 1", trips=["t1"], vehicle_type="maxi")}
    vehicle = _maxi_spec(battery_capacity_kwh=1.0, soc_min=0.0, soc_max=1.0)
    is_addable = build_default_is_addable(
        route_dict=route_dict,
        vehicle_capacity_map={"maxi": usable_battery_capacity_j(vehicle)},
    )
    mission = DrivingMission()
    mission.add_trip(
        Trip(
            id="existing",
            route="r1",
            stops=["a", "b"],
            arrival_times=["07:00:00", "07:10:00"],
            estimated_energy_demand=0.5e6,
        )
    )

    accepted_trip = Trip(
        id="t1",
        route="r1",
        stops=["b", "c"],
        arrival_times=["08:00:00", "08:10:00"],
        estimated_energy_demand=3.0e6,
    )
    rejected_trip = accepted_trip.model_copy(update={"id": "t2", "estimated_energy_demand": 4.0e6})

    assert is_addable(mission, accepted_trip)
    assert not is_addable(mission, rejected_trip)


def test_calculate_energy_demand_cache_key_includes_vehicle(monkeypatch):
    estimator = EnergyDemandEstimator.__new__(EnergyDemandEstimator)
    estimator.stop_dict = {}
    estimator.elevation_oracle = None
    estimator.cache = {}

    trip = Trip(
        id="t1",
        route="r1",
        stops=["a", "b"],
        arrival_times=["08:00:00", "08:10:00"],
    )
    geometry = TripGeometry(
        lon=[0.0, 1.0],
        lat=[0.0, 1.0],
        distance=[100.0],
        elevation=[0.0, 0.0],
        is_stop=[True, True],
        speed_limit=[50.0],
    )
    calls = []

    def fake_download_geometry(self, stop_dict, elevation_oracle=None):
        return geometry

    def fake_compute_speed_profile(itinerary, bus, comfort, payload=0.0, aux_power=0.0):
        calls.append((bus.chassis.aerodynamic_drag_area.m_as("m**2"), payload))
        return SpeedProfile(
            time=Quantity(np.array([0.0, 10.0]), "s"),
            distance=Quantity(np.array([0.0, 100.0]), "m"),
            speed=Quantity(np.array([0.0, 10.0]), "m/s"),
            acceleration=Quantity(np.array([1.0]), "m/s**2"),
            strategy_used=_CasadiStrategyTag(),
        )

    class FakeVehicleDynamics:
        traction_force = Quantity(np.array([100.0, 100.0]), "N")

    monkeypatch.setattr(Trip, "download_geometry", fake_download_geometry)
    monkeypatch.setattr(
        EnergyDemandEstimator, "_compute_speed_profile", staticmethod(fake_compute_speed_profile)
    )
    monkeypatch.setattr(
        "trip_stitcher.energy_demand_estimator.LongitudinalVehicleDynamics.of",
        lambda bus, mission: FakeVehicleDynamics(),
    )

    first_vehicle = _maxi_spec(aerodynamic_drag_area_m2=4.0, passenger_payload_kg=123.0)
    second_vehicle = _maxi_spec(aerodynamic_drag_area_m2=5.0, passenger_payload_kg=123.0)

    estimator.calculate_energy_demand(trip, vehicle=first_vehicle)
    estimator.calculate_energy_demand(trip, vehicle=second_vehicle)
    estimator.calculate_energy_demand(trip, vehicle=second_vehicle)

    assert calls == [(4.0, 123.0), (5.0, 123.0)]


def test_calculate_energy_demand_requires_vehicle():
    estimator = EnergyDemandEstimator.__new__(EnergyDemandEstimator)
    estimator.cache = {}
    trip = Trip(id="t1", route="r1", stops=["a", "b"], arrival_times=["08:00:00", "08:10:00"])

    with pytest.raises(ValueError, match="vehicle must not be None"):
        estimator.calculate_energy_demand(trip)
