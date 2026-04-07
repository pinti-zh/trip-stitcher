import sys
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import Levenshtein
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from sqlalchemy import MetaData, create_engine, select


def is_valid_yyyymmdd(date_str):
    try:
        # Try to parse the string as a date
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        # Raised if format is wrong or date is invalid
        return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--db-name", type=str, dest="db_name", default="gtfs")
    parser.add_argument("--query-output-file", type=str, dest="query_output_file")
    parser.add_argument("--chunk-size", type=int, dest="chunk_size", default=100000)
    parser.add_argument("--agency", type=str, dest="agency", default="801", help="agency id or name")
    parser.add_argument(
        "--date", type=str, dest="date", default="20250113", help="date of target day (format: YYYYMMDD)"
    )
    args = parser.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    engine = create_engine(f"sqlite:///{args.db_name}.db", future=True)
    logger.debug(f"created engine for {args.db_name}.db")
    metadata = MetaData()

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if not is_valid_yyyymmdd(args.date):
        logger.error(f"wrong date format: {args.date}, expected YYYYMMDD")
        return

    year, month, day = map(int, [args.date[:4], args.date[4:6], args.date[6:]])
    target_date_object = date(year, month, day)
    weekday = target_date_object.weekday()
    logger.debug(f"target date: {target_date_object} ({weekdays[weekday]})")

    metadata.reflect(engine)

    agency = metadata.tables["agency"]
    agency_stmt = select(agency.c.agency_id, agency.c.agency_name)
    agency_id_to_name = {}
    agency_name_to_id = {}
    with engine.connect() as conn:
        for result in conn.execute(agency_stmt):
            agency_id_to_name[result.agency_id] = result.agency_name
            agency_name_to_id[result.agency_name] = result.agency_id

    closest_agency_name = sorted(
        agency_name_to_id.keys(), key=lambda x: Levenshtein.distance(x.lower(), args.agency.lower()) / len(x)
    )[0]
    closest_agency_id = sorted(
        agency_id_to_name.keys(), key=lambda x: Levenshtein.distance(x.lower(), args.agency.lower()) / len(x)
    )[0]

    name_distance = Levenshtein.distance(closest_agency_name.lower(), args.agency.lower()) / len(closest_agency_name)
    id_distance = Levenshtein.distance(closest_agency_id.lower(), args.agency.lower()) / len(closest_agency_id)

    if name_distance < id_distance:
        agency_id = agency_name_to_id[closest_agency_name]
    else:
        agency_id = closest_agency_id

    logger.info(f"using agency {agency_id}: {agency_id_to_name[agency_id]}")

    calendar = metadata.tables["calendar"]
    calendar_dates = metadata.tables["calendar_dates"]

    weekday_fields = [
        calendar.c.monday,
        calendar.c.tuesday,
        calendar.c.wednesday,
        calendar.c.thursday,
        calendar.c.friday,
        calendar.c.saturday,
        calendar.c.sunday,
    ]

    # Combined query following GTFS rules:
    # (services in calendar AND NOT in calendar_dates as removed) OR (services in calendar_dates as added)
    valid_services_stmt = (
        select(calendar.c.service_id)
        .where(calendar.c.start_date <= args.date)
        .where(calendar.c.end_date >= args.date)
        .where(weekday_fields[weekday] == 1)
        .where(
            calendar.c.service_id.not_in(
                select(calendar_dates.c.service_id)
                .where(calendar_dates.c.date == args.date)
                .where(calendar_dates.c.exception_type == 2)
            )
        )
        .union(
            select(calendar_dates.c.service_id)
            .where(calendar_dates.c.date == args.date)
            .where(calendar_dates.c.exception_type == 1)
        )
    ).subquery()

    trips = metadata.tables["trips"]
    stop_times = metadata.tables["stop_times"]
    stops = metadata.tables["stops"]
    routes = metadata.tables["routes"]
    query = (
        select(
            routes.c.agency_id,
            routes.c.route_id,
            routes.c.route_short_name,
            trips.c.trip_id,
            trips.c.service_id,
            stop_times.c.stop_sequence,
            stop_times.c.arrival_time,
            stop_times.c.departure_time,
            stops.c.stop_id,
            stops.c.stop_name,
            stops.c.stop_lat,
            stops.c.stop_lon,
        )
        .where(routes.c.agency_id == agency_id)
        .where(trips.c.service_id.in_(select(valid_services_stmt)))
        .join(trips, trips.c.route_id == routes.c.route_id)
        .join(stop_times, stop_times.c.trip_id == trips.c.trip_id)
        .join(stops, stops.c.stop_id == stop_times.c.stop_id)
    )

    output_file_path = Path(args.query_output_file)

    with engine.connect() as conn:
        logger.info(f"running aggregation query and writing to {output_file_path} ...")
        query_start_time = perf_counter()
        writer = None
        for chunk in pd.read_sql_query(query, conn, chunksize=args.chunk_size):
            table = pa.Table.from_pandas(df=chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    where=output_file_path,
                    schema=table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

        if writer is not None:
            writer.close()
        query_time = perf_counter() - query_start_time
        logger.info(f"  done (query time = {query_time:.2f} seconds)")

    num_rows = pq.ParquetFile(output_file_path).metadata.num_rows
    logger.info(f"written {num_rows} rows to file")


if __name__ == "__main__":
    main()
