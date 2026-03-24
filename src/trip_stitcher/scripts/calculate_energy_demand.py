from functools import partial

import pandas as pd
from loguru import logger

from trip_stitcher.energy_demand_estimator import EnergyDemandEstimator
from trip_stitcher.models import Stop, Trip, TripGeometry
from trip_stitcher.pipeline import get_default_parser, run_pipeline
from trip_stitcher.utils import setup_logger


def calculate_energy_demand(
    trip: Trip,
    ed_estimator: EnergyDemandEstimator | None = None,
    bus_type=None,
    df: pd.DataFrame | None = None,
    aux_power: float = 0.0,
) -> Trip:
    if df is None:
        raise ValueError("df must not be None")
    if ed_estimator is None:
        raise ValueError("ed_estimator must not be None")
    energy_demand = ed_estimator.calculate_energy_demand(trip, bus_type=bus_type, aux_power=aux_power)
    trip.estimated_energy_demand = energy_demand
    return trip


def main():
    parser = get_default_parser()
    parser.add_argument("--bus-type", choices=["mini", "maxi", "mega"], default="maxi", help="bus type")
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.data_file)
    setup_logger(args.debug)
    ed_estimator = EnergyDemandEstimator(df)
    run_pipeline(
        partial(calculate_energy_demand, ed_estimator=ed_estimator, bus_type=args.bus_type, df=df), Trip, args
    )


if __name__ == "__main__":
    main()
