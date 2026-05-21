import csv
import os
from argparse import ArgumentParser
from pathlib import Path

from sqlalchemy import MetaData, create_engine, text


def export_db_to_gtfs(db_url: str, output_dir: str):
    """
    Exports all tables from a SQLite/Postgres database back into standard GTFS .txt files.
    """
    # 1. Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Bind engine and reflect the schema to discover what tables exist
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    print(f"Found {len(metadata.tables)} tables to export.\n")

    # 3. Stream each table directly into its respective text file
    with engine.connect() as conn:
        for table_name in metadata.tables.keys():
            # GTFS file naming standard
            file_name = f"{table_name}.txt"
            file_path = os.path.join(output_dir, file_name)

            print(f"Exporting '{table_name}' to {file_name}...")

            # Execute query to pull all filtered rows
            result = conn.execute(text(f"SELECT * FROM {table_name}"))

            # Extract header names directly from the query execution cursor
            headers = list(result.keys())

            if not headers:
                print(f"  Skipping '{table_name}' — table has no columns.")
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

            print(f"  Successfully wrote {file_name}")

    print("\nAll tables exported successfully back to text format!")

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

    # Your small, filtered database file path
    SMALL_DB = "sqlite:///small_local_gtfs.db"

    # Target directory where you want your filtered GTFS files to live
    OUTPUT_FOLDER = "./filtered_gtfs_feed"

    export_db_to_gtfs(db_url=SMALL_DB, output_dir=OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
