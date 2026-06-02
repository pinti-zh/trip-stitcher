import csv
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger
from sqlalchemy import MetaData, create_engine, text

TABLES = [
    "agency",
    "routes",
    "trips",
    "stop_times",
    "stops",
    "calendar",
    "calendar_dates",
    "transfers",
]


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--input-db", type=Path, dest="input_db", required=True, help="path to input database"
    )
    parser.add_argument(
        "--output-dir", type=Path, dest="output_dir", required=True, help="path to output directory"
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

    engine = create_engine(f"sqlite:///{args.input_db}")
    metadata = MetaData()
    metadata.reflect(bind=engine)

    assert all(table in metadata.tables.keys() for table in TABLES)
    logger.debug("all tables found")

    if not args.output_dir.exists():
        os.makedirs(args.output_dir)

    # 3. Stream each table directly into its respective text file
    with engine.connect() as conn:
        for table_name in TABLES:
            # GTFS file naming standard
            file_name = f"{table_name}.txt"
            file_path = args.output_dir / file_name

            logger.debug(f"exporting '{table_name}' to {file_name}...")

            # Execute query to pull all filtered rows
            result = conn.execute(text(f"SELECT * FROM {table_name}"))

            # Extract header names directly from the query execution cursor
            headers = list(result.keys())

            if not headers:
                logger.warning(f"  Skipping '{table_name}' — table has no columns.")
                continue

            # Open file and write content using standard CSV formats required by GTFS spec
            # lineterminator='\r\n' is strictly preferred by the GTFS specification
            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")

                # Write the header row
                writer.writerow(headers)

                # Stream row chunks to safely handle tables with high row counts (like stop_times)
                # without destroying system memory
                while True:
                    rows = result.fetchmany(25000)
                    if not rows:
                        break

                    # Write rows directly to file block by block
                    writer.writerows(rows)

            logger.debug(f"  successfully wrote {file_name}")

    logger.info(f"all tables exported successfully back to text format and saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
