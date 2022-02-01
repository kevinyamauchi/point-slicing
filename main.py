from pathlib import Path

import zarr
from numpy.random import default_rng
from zarr.errors import PathNotFoundError

from utils.cli import read_args
from utils.timer import timer
from utils.vars import DTYPE, RANDOM_SEED

_file_location = Path(__file__).resolve()


@timer
def create_zarr_random_points(
    ndim: int, points_per_dim: int, chunk_size: int
) -> zarr.Array:
    """
    creates a d-dimensional zarr dataset of randomly distributed points
    """
    filepath = (
        _file_location.parent
        / "data"
        / f"random_points_dim{ndim}_points{points_per_dim}.zarr"
    )
    box_shape = (points_per_dim,) * ndim
    try:
        box = zarr.open(filepath, mode="r")
    except PathNotFoundError:
        box = zarr.open(
            filepath,
            mode="w",
            shape=box_shape,
            chunks=(chunk_size,) * ndim,
            dtype=DTYPE,
        )
        rng = default_rng(RANDOM_SEED)
        box[:] = rng.random(box_shape)
    return box


def find_2d_slice_of_3d_box(box: zarr.Array, x: int, y: int, z: int) -> zarr.Array:
    """
    Finds a 2d slice of a 3d points using linear search by calculating
    distance to the slicing plane for each point. Points within 0.5 units
    of the slice are classed as belonging to the slice.
    """
    return box[x, y, z]


if __name__ == "__main__":
    args = read_args()
    box = create_zarr_random_points(args.ndim, args.points, args.chunksize)
    find_2d_slice_of_3d_box(box, args.x, args.y, args.z)
