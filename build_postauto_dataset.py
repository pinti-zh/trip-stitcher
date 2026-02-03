import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from sqlalchemy import MetaData, create_engine, select


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--db-name", type=str, dest="db_name", default="gtfs")
    parser.add_argument("--query-output-file", type=str, dest="query_output_file")
    parser.add_argument("--chunk-size", type=int, dest="chunk_size", default=100000)
    args = parser.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    engine = create_engine(f"sqlite:///{args.db_name}.db", future=True)
    logger.debug(f"created engine for {args.db_name}.db")
    metadata = MetaData()

    agency_id = 801 # PostAuto AG

    metadata.reflect(engine)
    trips = metadata.tables["trips"]
    stop_times = metadata.tables["stop_times"]
    stops = metadata.tables["stops"]
    routes = metadata.tables["routes"]
    calendar = metadata.tables["calendar"]

    query = (
        select(
            routes.c.agency_id,
            routes.c.route_id,
            routes.c.route_short_name,
            trips.c.trip_id,
            trips.c.service_id,
            calendar.c.monday,
            calendar.c.start_date,
            calendar.c.end_date,
            stop_times.c.stop_sequence,
            stop_times.c.arrival_time,
            stop_times.c.departure_time,
            stops.c.stop_id,
            stops.c.stop_name,
            stops.c.stop_lat,
            stops.c.stop_lon,
        )
        .where(routes.c.agency_id == agency_id)
        .join(trips, trips.c.route_id == routes.c.route_id)
        .join(stop_times, stop_times.c.trip_id == trips.c.trip_id)
        .join(stops, stops.c.stop_id == stop_times.c.stop_id)
        .join(calendar, calendar.c.service_id == trips.c.service_id)
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
