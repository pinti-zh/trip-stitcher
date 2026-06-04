import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from loguru import logger


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument("--gtfs-dir", type=Path, required=True, help="Path to GTFS directory")
    args = parser.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    if not args.gtfs_dir.exists():
        logger.error(f"GTFS directory '{args.gtfs_dir}' does not exist")
        sys.exit(1)

    for file_path in ["shapes.txt", "trips.txt"]:
        if not (args.gtfs_dir / file_path).exists():
            logger.error(f"'{file_path}' does not exist")
            sys.exit(1)

    shapes = pd.read_csv(args.gtfs_dir / "shapes.txt")
    num_shape_rows = len(shapes)
    logger.debug(f"loaded shapes ({num_shape_rows} rows)")

    trips = pd.read_csv(args.gtfs_dir / "trips.txt")
    logger.debug(f"loaded trips ({len(trips)} rows)")

    id_map: dict[str, set[str]] = {}  # maps each shape_id to a set of partial trip_ids
    new_shape_ids = []
    for trip_id, shape_id in zip(trips["trip_id"], trips["shape_id"]):
        split_trip_id = trip_id.split(".")
        assert len(split_trip_id) == 5
        partial_trip_id = ".".join(split_trip_id[-3:])
        if shape_id not in id_map:
            id_map[shape_id] = {partial_trip_id}
        else:
            id_map[shape_id].add(partial_trip_id)
        new_shape_ids.append(partial_trip_id)

    assert all(shape_id in id_map.keys() for shape_id in shapes["shape_id"])

    min_set_size, max_set_size = min(map(len, id_map.values())), max(map(len, id_map.values()))
    logger.debug(f"id_map maps shape_ids to sets of size {min_set_size} to {max_set_size}")

    shapes["shape_id"] = shapes["shape_id"].map(id_map)
    shapes = shapes.explode("shape_id", ignore_index=True)
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)
    assert min_set_size * num_shape_rows <= len(shapes) <= max_set_size * num_shape_rows
    logger.debug(f"converted shapes dataframe contains {len(shapes)} rows")

    shapes.to_csv(args.gtfs_dir / "shapes.txt", index=False)
    logger.info(f"written shapes to '{args.gtfs_dir / 'shapes.txt'}'")
    
    trips["shape_id"] = new_shape_ids
    trips.to_csv(args.gtfs_dir / "trips.txt", index=False)
    logger.info(f"written trips to '{args.gtfs_dir / 'trips.txt'}'")


if __name__ == "__main__":
    main()
