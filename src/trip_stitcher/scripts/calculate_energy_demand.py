# calculate_energy_demand.py
#
# Pipeline script that reads Trip objects from a JSONL input stream, estimates
# the energy demand for each trip using the EnergyDemandEstimator, and writes
# the annotated Trip objects to a JSONL output stream.
#
# The script relies on a pre-built elevation/speed Parquet data file and the
# trip_stitcher pipeline infrastructure (get_default_parser / run_pipeline).
#
# Usage:
#   python calculate_energy_demand.py --data-file FILE
#                                     [--bus-type {mini,maxi,mega}]
#                                     [--input FILE] [--output FILE] [--debug]
#
# Arguments:
#   --data-file   Path to the Parquet file used by EnergyDemandEstimator
#   --bus-type    Vehicle type to use for the simulation (default: maxi)
#   --debug       Enable verbose DEBUG-level logging

from functools import partial

import pandas as pd

from trip_stitcher.energy_demand_estimator import EnergyDemandEstimator
from trip_stitcher.models import Trip
from trip_stitcher.pipeline import get_default_parser, run_pipeline
from trip_stitcher.utils import setup_logger
from trip_stitcher.vehicle_specs import VehicleSpec, VehicleType
from trip_stitcher.vehicles import maxi, mega, mini

_BUILTIN_BUSES = {
    "mini": mini.bus,
    "maxi": maxi.bus,
    "mega": mega.bus,
}


def calculate_energy_demand(
    trip: Trip,
    ed_estimator: EnergyDemandEstimator | None = None,
    vehicle: VehicleSpec | None = None,
    df: pd.DataFrame | None = None,
    aux_power: float = 0.0,
) -> Trip:
    if df is None:
        raise ValueError("df must not be None")
    if ed_estimator is None:
        raise ValueError("ed_estimator must not be None")
    if vehicle is None:
        raise ValueError("vehicle must not be None")
    energy_demand = ed_estimator.calculate_energy_demand(trip, aux_power=aux_power, vehicle=vehicle)
    trip.estimated_energy_demand = energy_demand
    return trip


def _vehicle_spec_from_builtin(base_type: VehicleType) -> VehicleSpec:
    bus = _BUILTIN_BUSES[base_type]
    return VehicleSpec(
        base_type=base_type,
        curb_weight_kg=bus.curb_weight.m_as("kg"),
        passenger_payload_kg=0.0,
        aerodynamic_drag_area_m2=bus.chassis.aerodynamic_drag_area.m_as("m**2"),
        rolling_friction_coefficient=bus.chassis.rolling_friction,
        battery_capacity_kwh=bus.battery.capacity.m_as("kWh"),
        soc_min=bus.battery.soc_min,
        soc_max=bus.battery.soc_max,
    )


def main():
    parser = get_default_parser()
    parser.add_argument(
        "--bus-type", choices=["mini", "maxi", "mega"], default="maxi", help="bus type"
    )
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.data_file)
    setup_logger(args.debug)
    ed_estimator = EnergyDemandEstimator(df)
    vehicle = _vehicle_spec_from_builtin(args.bus_type)
    run_pipeline(
        partial(calculate_energy_demand, ed_estimator=ed_estimator, vehicle=vehicle, df=df),
        Trip,
        args,
    )


if __name__ == "__main__":
    main()
