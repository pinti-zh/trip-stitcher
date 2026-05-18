import os
import sys
from argparse import ArgumentParser, FileType, Namespace
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


def get_default_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--input", type=FileType("r"), default=sys.stdin, help="Input file (defaults to stdin)"
    )
    parser.add_argument(
        "--output", type=FileType("w"), default=sys.stdout, help="Output file (defaults to stdout)"
    )
    return parser


def collect(input_cls: Type[InputModel], args: Namespace) -> list[InputModel]:
    data_collection: list[InputModel] = []
    for line in args.input:
        line = line.strip()
        if not line:
            continue
        data = input_cls.parse_raw(line)
        assert isinstance(data, input_cls)
        data_collection.append(data)
    return data_collection


def run_pipeline(
    transform: Callable[[InputModel], OutputModel | list[OutputModel]],
    input_cls: Type[InputModel],
    args: Namespace,
):
    for line in args.input:
        line = line.strip()
        if not line:
            continue
        data = input_cls.model_validate_json(line)
        transformed_data = transform(data)
        if transformed_data is not None:
            try:
                if isinstance(transformed_data, list):
                    for item in transformed_data:
                        assert isinstance(item, BaseModel)
                        print(item.model_dump_json(), file=args.output, flush=True)
                else:
                    print(transformed_data.model_dump_json(), file=args.output, flush=True)
            except BrokenPipeError:
                os._exit(0)
