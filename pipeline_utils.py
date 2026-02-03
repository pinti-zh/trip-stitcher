import os
import sys
from argparse import ArgumentParser, FileType, Namespace
from contextlib import contextmanager
from typing import Callable, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


def setup_logger(debug: bool):
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def get_default_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input", type=FileType("r"), default=sys.stdin, help="Input file (defaults to stdin)")
    parser.add_argument("--output", type=FileType("w"), default=sys.stdout, help="Output file (defaults to stdout)")
    return parser


def run_generator(generator: Callable[..., BaseModel], args: Namespace):
    for data in generator():
        if data is not None:
            try:
                assert isinstance(data, BaseModel)
                print(data.json(), file=args.output, flush=True)
            except BrokenPipeError:
                os._exit(0)


def run_pipeline(transform: Callable[[InputModel], OutputModel], input_cls: Type[InputModel], args: Namespace):
    for line in args.input:
        line = line.strip()
        if not line:
            continue
        data = input_cls.parse_raw(line)
        transformed_data = transform(data)
        if transformed_data is not None:
            try:
                print(transformed_data.json(), file=args.output, flush=True)
            except BrokenPipeError:
                os._exit(0)
