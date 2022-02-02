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
    DEFAULT_TOL,
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
        "--alpha",
        "-a",
        type=int,
        default=DEFAULT_ALPHA,
        help=f"Euler angle alpha in degrees in range [0,360), default: {DEFAULT_ALPHA}",
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=int,
        default=DEFAULT_BETA,
        help=f"Euler angle beta in degrees in range [0,180], default: {DEFAULT_BETA}",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=int,
        default=DEFAULT_GAMMA,
        help=f"Euler angle gamma in degrees in range [0,360), default: {DEFAULT_GAMMA}",
    )
    parser.add_argument(
        "-x",
        type=float,
        default=DEFAULT_X,
        help=f"x-coordinate to rotate about, default: {DEFAULT_X}",
    )
    parser.add_argument(
        "-y",
        type=float,
        default=DEFAULT_Y,
        help=f"y-coordinate to rotate about, default: {DEFAULT_Y}",
    )
    parser.add_argument(
        "-z",
        type=float,
        default=DEFAULT_Z,
        help=f"z-coordinate to rotate about, default: {DEFAULT_Z}",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=DEFAULT_TOL,
        help=f"z-coordinate to rotate about, default: {DEFAULT_Z}",
    )
    args = parser.parse_args()
    logger.info(
        f"dimensions: {args.ndim}, "
        f"points per dimensions: {args.points}, "
        f"chunk size: {args.chunksize}, "
        f"number of bytes {np.dtype(DTYPE).itemsize*args.points*args.ndim:e}, "
        f"alpha: {args.alpha}\N{DEGREE SIGN}, "
        f"beta: {args.beta}\N{DEGREE SIGN}, "
        f"gamma: {args.gamma}\N{DEGREE SIGN}, "
        f"x: {args.x}, "
        f"y: {args.y}, "
        f"z: {args.z}, "
    )
    return args
