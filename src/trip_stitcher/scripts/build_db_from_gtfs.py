import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    inspect,
    select,
)


def load_gtfs_table(engine: Engine, table: Table, file_path: Path) -> None:
    df = pd.read_csv(file_path, dtype=str)
    df.where(pd.notnull(df), None, inplace=True)  # NaN → NULL

    with engine.begin() as conn:
        conn.execute(table.insert(), df.to_dict(orient="records"))


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--gtfs-directory", type=str, dest="gtfs_directory", default="data/2025_google_transit"
    )
    parser.add_argument("--db-name", type=str, dest="db_name", default="gtfs")
    args = parser.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # --- Connect ---
    engine = create_engine(f"sqlite:///{args.db_name}.db", future=True)
    metadata = MetaData()

    # --- Tables ---
    agency = Table(
        "agency",
        metadata,
        Column("agency_id", Text, primary_key=True),
        Column("agency_name", Text),
        Column("agency_url", Text),
        Column("agency_timezone", Text),
    )

    routes = Table(
        "routes",
        metadata,
        Column("route_id", Text, primary_key=True),
        Column("agency_id", Text),
        Column("route_short_name", Text),
        Column("route_long_name", Text),
        Column("route_type", Integer),
    )

    trips = Table(
        "trips",
        metadata,
        Column("trip_id", Text, primary_key=True),
        Column("route_id", Text),
        Column("service_id", Text),
        Column("trip_headsign", Text),
        Column("direction_id", Integer),
        Column("shape_id", Text),
    )

    stops = Table(
        "stops",
        metadata,
        Column("stop_id", Text, primary_key=True),
        Column("stop_name", Text),
        Column("stop_lat", Float),
        Column("stop_lon", Float),
    )

    stop_times = Table(
        "stop_times",
        metadata,
        Column("trip_id", Text),
        Column("arrival_time", Text),
        Column("departure_time", Text),
        Column("stop_id", Text),
        Column("stop_sequence", Integer),
    )

    calendar = Table(
        "calendar",
        metadata,
        Column("service_id", Text, primary_key=True),
        Column("monday", Integer),
        Column("tuesday", Integer),
        Column("wednesday", Integer),
        Column("thursday", Integer),
        Column("friday", Integer),
        Column("saturday", Integer),
        Column("sunday", Integer),
        Column("start_date", Text),
        Column("end_date", Text),
    )

    calendar_dates = Table(
        "calendar_dates",
        metadata,
        Column("service_id", Text),
        Column("date", Text),
        Column("exception_type", Integer),
    )

    shapes = Table(
        "shapes",
        metadata,
        Column("shape_id", Text, nullable=False),
        Column("shape_pt_lat", Float),
        Column("shape_pt_lon", Float),
        Column("shape_pt_sequence", Integer, nullable=False),
        Column("shape_dist_traveled", Float),
    )

    transfers = Table(
        "transfers",
        metadata,
        Column("from_stop_id", Text, nullable=False),
        Column("to_stop_id", Text, nullable=False),
        Column("transfer_type", Integer),
        Column("min_transfer_time", Integer),
    )

    # --- Indices ---
    Index("idx_stop_times_trip_id", stop_times.c.trip_id)
    Index("idx_stop_times_stop_id", stop_times.c.stop_id)
    Index("idx_trips_route_id", trips.c.route_id)
    Index("idx_trips_service_id", trips.c.service_id)

    tables = {
        "agency": agency,
        "routes": routes,
        "trips": trips,
        "stops": stops,
        "stop_times": stop_times,
        "calendar": calendar,
        "calendar_dates": calendar_dates,
        "shapes": shapes,
        "transfers": transfers,
    }

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    gtfs_dir = Path(args.gtfs_directory)  # folder with agency.txt, routes.txt, etc.

    if not existing_tables:
        logger.info("creating tables and loading GTFS...")
        metadata.create_all(engine)
        for name, table in tables.items():
            path = gtfs_dir / f"{name}.txt"
            if path.exists():
                logger.info(f"loading {name}")
                load_gtfs_table(engine, table, path)
            else:
                logger.warning(f"skipping {name} (missing)")
    else:
        logger.warning("database already exists — skipping table creation and data load")

    stmt = (
        select(routes.c.route_short_name, stops.c.stop_name, stop_times.c.arrival_time)
        .select_from(stop_times)
        .join(trips, stop_times.c.trip_id == trips.c.trip_id)
        .join(routes, trips.c.route_id == routes.c.route_id)
        .join(stops, stop_times.c.stop_id == stops.c.stop_id)
        .limit(5)
    )

    start = perf_counter()
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    query_time = perf_counter() - start
    logger.info(f"test query result: ({query_time:.3f} seconds)")
    for r in rows:
        logger.info(r)


if __name__ == "__main__":
    main()
