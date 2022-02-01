from pathlib import Path

import dask.array as da
import zarr

from utils.cli import read_args
from utils.timer import timer

_file_location = Path(__file__).resolve()


@timer
def create_zarr_random_points(
    ndim: int, points_per_dim: int, chunk_size: int
) -> zarr.Array:
    """
    creates a d-dimensional zarr dataset of radnomly distributed points
    """
    filepath = (
        _file_location.parent
        / "data"
        / f"random_points_dim{ndim}_points{points_per_dim}.zarr"
    )
    if filepath.exists():
        box = zarr.load(filepath)
    else:
        box = da.random.random(
            size=(points_per_dim,) * ndim, chunks=(chunk_size,) * ndim
        ).to_zarr(filepath)
    return box


if __name__ == "__main__":
    args = read_args()
    z = create_zarr_random_points(args.ndim, args.points, args.chunksize)
