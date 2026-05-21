"""
This script filters a GTFS SQLite database by a specific agency.
It extracts all relevant records (routes, trips, stop times, stops, calendar, and calendar dates) 
associated with the selected agency and saves them to a new database.

Arguments:
    --debug: Enable debug logging.
    --input-db: Path to the input SQLite database.
    --output-db: Path to the output SQLite database.
    --agency: Agency ID or name to filter by (default: "801").
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger
from sqlalchemy import MetaData, bindparam, create_engine, text

from trip_stitcher.utils import match_agency_string


def main():
    """
    Parses command-line arguments, connects to source and destination databases,
    and performs the filtering and insertion process.
    """
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--input-db", type=Path, dest="input_db", required=True, help="path to input database"
    )
    parser.add_argument(
        "--output-db", type=Path, dest="output_db", required=True, help="path to output database"
    )
    parser.add_argument(
        "--agency", type=str, dest="agency", default="801", help="agency id or name to filter by"
    )
    args = parser.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    if not args.input_db.exists():
        logger.error(f"input_db does not exist: {args.input_db}")
        sys.exit(1)

    if args.output_db.exists():
        os.remove(args.output_db)

    metadata = MetaData()

    src_engine = create_engine(f"sqlite:///{args.input_db}")
    metadata.reflect(bind=src_engine)

    dst_engine = create_engine(f"sqlite:///{args.output_db}")
    metadata.create_all(bind=dst_engine)

    agency_id, agency_name = match_agency_string(args.agency, src_engine, metadata)
    logger.debug(f"matched input agency '{args.agency}' to ({agency_id}, {agency_name})")

    # Open standard modern connections on both ends
    with src_engine.connect() as src_conn:
        with dst_engine.connect() as dst_conn:
            with dst_conn.begin():
                # Optimize SQLite insertion performance
                if dst_engine.dialect.name == "sqlite":
                    dst_conn.execute(text("PRAGMA synchronous = OFF;"))
                    dst_conn.execute(text("PRAGMA journal_mode = MEMORY;"))

                # 1. Filter 'agency' table
                agency_row_count = src_conn.execute(text("SELECT COUNT(*) FROM agency")).scalar()
                q_agency = text("SELECT * FROM agency WHERE agency_id = :agency_id")
                rows_agency_filtered = src_conn.execute(
                    q_agency, {"agency_id": agency_id}
                ).fetchall()
                logger.debug(
                    f"filtered agency: {len(rows_agency_filtered)}/{agency_row_count} row(s) remaining"
                )

                # 2. Filter 'routes' based on agency_id
                routes_row_count = src_conn.execute(text("SELECT COUNT(*) FROM routes")).scalar()
                q_routes = text("SELECT * FROM routes WHERE agency_id = :agency_id")
                rows_routes_filtered = src_conn.execute(
                    q_routes, {"agency_id": agency_id}
                ).fetchall()
                logger.debug(
                    f"filtered routes: {len(rows_routes_filtered)}/{routes_row_count} row(s) remaining"
                )

                # 3. Filter 'trips' based on the filtered route_ids
                trips_row_count = src_conn.execute(text("SELECT COUNT(*) FROM trips")).scalar()
                q_trips = text("SELECT * FROM trips WHERE route_id IN :route_ids").bindparams(
                    bindparam("route_ids", expanding=True)
                )
                rows_trips_filtered = src_conn.execute(
                    q_trips, {"route_ids": [row.route_id for row in rows_routes_filtered]}
                ).fetchall()
                logger.debug(
                    f"filtered trips: {len(rows_trips_filtered)}/{trips_row_count} row(s) remaining"
                )

                # 4. Filter 'stop_times' based on the filtered trip_ids
                stop_times_row_count = src_conn.execute(
                    text("SELECT COUNT(*) FROM stop_times")
                ).scalar()
                q_stop_times = text(
                    "SELECT * FROM stop_times WHERE trip_id IN :trip_ids"
                ).bindparams(bindparam("trip_ids", expanding=True))
                rows_stop_times_filtered = src_conn.execute(
                    q_stop_times, {"trip_ids": [row.trip_id for row in rows_trips_filtered]}
                ).fetchall()

                logger.debug(
                    f"filtered stop_times: {len(rows_stop_times_filtered)}/{stop_times_row_count} row(s) remaining"
                )

                # 5. Filter 'stops' based on stop_ids used in filtered stop_times
                stops_row_count = src_conn.execute(text("SELECT COUNT(*) FROM stops")).scalar()
                q_stops = text("SELECT * FROM stops WHERE stop_id IN :stop_ids").bindparams(
                    bindparam("stop_ids", expanding=True)
                )
                rows_stops_filtered = src_conn.execute(
                    q_stops,
                    {"stop_ids": list(set(row.stop_id for row in rows_stop_times_filtered))},
                ).fetchall()

                logger.debug(
                    f"filtered stops: {len(rows_stops_filtered)}/{stops_row_count} row(s) remaining"
                )

                # 6. Filter 'calendar' based on service_ids used in filtered trips
                calendar_row_count = src_conn.execute(
                    text("SELECT COUNT(*) FROM calendar")
                ).scalar()
                q_calendar = text(
                    "SELECT * FROM calendar WHERE service_id IN :service_ids"
                ).bindparams(bindparam("service_ids", expanding=True))
                rows_calendar_filtered = src_conn.execute(
                    q_calendar,
                    {"service_ids": list(set(row.service_id for row in rows_trips_filtered))},
                ).fetchall()

                logger.debug(
                    f"filtered calendar: {len(rows_calendar_filtered)}/{calendar_row_count} row(s) remaining"
                )

                # 7. Filter 'calendar_dates' based on service_ids used in filtered trips
                calendar_dates_row_count = src_conn.execute(
                    text("SELECT COUNT(*) FROM calendar_dates")
                ).scalar()
                q_calendar_dates = text(
                    "SELECT * FROM calendar_dates WHERE service_id IN :service_ids"
                ).bindparams(bindparam("service_ids", expanding=True))
                rows_calendar_dates_filtered = src_conn.execute(
                    q_calendar_dates,
                    {"service_ids": list(set(row.service_id for row in rows_trips_filtered))},
                ).fetchall()

                logger.debug(
                    f"filtered calendar_dates: {len(rows_calendar_dates_filtered)}/{calendar_dates_row_count} row(s) remaining"
                )

                # Insert filtered data into the destination database
                dst_conn.execute(
                    metadata.tables["agency"].insert(),
                    [row._asdict() for row in rows_agency_filtered],
                )
                dst_conn.execute(
                    metadata.tables["routes"].insert(),
                    [row._asdict() for row in rows_routes_filtered],
                )
                dst_conn.execute(
                    metadata.tables["trips"].insert(),
                    [row._asdict() for row in rows_trips_filtered],
                )
                dst_conn.execute(
                    metadata.tables["stop_times"].insert(),
                    [row._asdict() for row in rows_stop_times_filtered],
                )
                dst_conn.execute(
                    metadata.tables["stops"].insert(),
                    [row._asdict() for row in rows_stops_filtered],
                )
                dst_conn.execute(
                    metadata.tables["calendar"].insert(),
                    [row._asdict() for row in rows_calendar_filtered],
                )
                dst_conn.execute(
                    metadata.tables["calendar_dates"].insert(),
                    [row._asdict() for row in rows_calendar_dates_filtered],
                )
                logger.info(f"written filtered data to {args.output_db}")


if __name__ == "__main__":
    main()
