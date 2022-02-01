from argparse import ArgumentParser, Namespace

import numpy as np

from utils.logger import logger
from utils.vars import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NDIM,
    DEFAULT_POINTS_PER_DIM,
    DEFAULT_X,
    DEFAULT_Y,
    DEFAULT_Z,
    DTYPE,
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
        help=f"zarr chunk size, default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--ndim",
        "-d",
        type=int,
        default=DEFAULT_NDIM,
        help=f"box dimensions, default: {DEFAULT_NDIM}",
    )
    parser.add_argument(
        "--points",
        "-p",
        type=int,
        default=DEFAULT_POINTS_PER_DIM,
        help=f"points per dimension, default: {DEFAULT_POINTS_PER_DIM}",
    )
    parser.add_argument(
        "-x",
        type=int,
        default=DEFAULT_X,
        help=f"Euler angle of x-coordinate, default: {DEFAULT_X}",
    )
    parser.add_argument(
        "-y",
        type=int,
        default=DEFAULT_Y,
        help=f"Euler angle of y-coordinate default: {DEFAULT_Y}",
    )
    parser.add_argument(
        "-z",
        type=int,
        default=DEFAULT_Z,
        help=f"Euler angle of z-coordinate, default: {DEFAULT_Z}",
    )
    args = parser.parse_args()
    logger.info(
        f"dimensions: {args.ndim}, "
        f"points per dimensions: {args.points}, "
        f"chunk size: {args.chunksize}, "
        f"number of bytes {np.dtype(DTYPE).itemsize*args.points**args.ndim:e}"
    )
    return args
