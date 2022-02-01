from argparse import ArgumentParser, Namespace

from utils.logger import logger
from utils.vars import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NDIM,
    DEFAULT_POINTS_PER_DIM,
    FLOAT64_BYTES,
)


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create zarr box")
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"zarr chunk size: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--ndim",
        "-d",
        type=int,
        default=DEFAULT_NDIM,
        help=f"box dimensions: {DEFAULT_NDIM}",
    )
    parser.add_argument(
        "--points",
        "-p",
        type=int,
        default=DEFAULT_POINTS_PER_DIM,
        help=f"points per dimension: {DEFAULT_POINTS_PER_DIM}",
    )
    args = parser.parse_args()
    logger.info(
        f"dimensions: {args.ndim}, "
        f"points per dimensions: {args.points}, "
        f"chunk size: {args.chunksize}, "
        f"requested number of bytes {FLOAT64_BYTES*args.points**args.ndim:e}"
    )
    return args
