from argparse import ArgumentParser, Namespace

import numpy as np

from utils.logger import logger
from utils.vars import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GAMMA,
    DEFAULT_NDIM,
    DEFAULT_POINTS_PER_DIM,
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
        "-a",
        type=int,
        default=DEFAULT_ALPHA,
        help=f"Euler angle alpha in degrees in range [0,360), default: {DEFAULT_ALPHA}",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=DEFAULT_BETA,
        help=f"Euler angle beta in degrees in range [0,180], default: {DEFAULT_BETA}",
    )
    parser.add_argument(
        "-g",
        type=int,
        default=DEFAULT_GAMMA,
        help=f"Euler angle gamma in degrees in range [0,360), default: {DEFAULT_GAMMA}",
    )
    parser.add_argument(
        "--points",
        "-p",
        type=int,
        default=DEFAULT_POINTS_PER_DIM,
        help=f"points per dimension, default: {DEFAULT_POINTS_PER_DIM}",
    )
    args = parser.parse_args()
    logger.info(
        f"dimensions: {args.ndim}, "
        f"points per dimensions: {args.points}, "
        f"chunk size: {args.chunksize}, "
        f"number of bytes {np.dtype(DTYPE).itemsize*args.points**args.ndim:e}"
    )
    return args
