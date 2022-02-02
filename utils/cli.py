from argparse import ArgumentParser, Namespace

import numpy as np

from utils.logger import logger
from utils.vars import DEFAULT_CHUNK_SIZE, DEFAULT_NDIM, DEFAULT_POINTS_PER_DIM, DTYPE


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
    args = parser.parse_args()
    logger.info(
        f"dimensions: {args.ndim}, "
        f"points per dimensions: {args.points}, "
        f"chunk size: {args.chunksize}, "
        f"number of bytes {np.dtype(DTYPE).itemsize*args.points*args.ndim:e}"
    )
    return args
