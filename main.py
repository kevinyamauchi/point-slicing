from pathlib import Path

import zarr
from numpy.random import default_rng
from zarr.errors import PathNotFoundError

from utils.cli import read_args
from utils.timer import timer
from utils.vars import DTYPE, RANDOM_SEED

_file_location = Path(__file__).resolve()


class ZarrExperiment:
    def __init__(self, chunk_size: int, ndim: int, points_per_dim: int) -> None:
        self.chunk_size = chunk_size
        self.ndim = ndim
        self.points_per_dim = points_per_dim
        self._data_path = (
            _file_location.parent
            / "data"
            / f"random_points_dim{self.ndim}_points{self.points_per_dim}.zarr"
        )
        self._rng = default_rng(RANDOM_SEED)
        self._create_zarr_random_points()

    @timer
    def _create_zarr_random_points(self) -> None:
        """
        creates a d-dimensional zarr dataset of randomly distributed points
        """
        box_shape = (self.points_per_dim,) * self.ndim
        try:
            self.box = zarr.open(self._data_path, mode="r")
        except PathNotFoundError:
            self.box = zarr.open(
                self._data_path,
                mode="w",
                shape=box_shape,
                chunks=(self.chunk_size,) * self.ndim,
                dtype=DTYPE,
            )
            self.box[:] = self._rng.random(box_shape)

    def find_2d_slice_of_3d_box(self, x: int, y: int, z: int) -> zarr.Array:
        """
        Finds a 2d slice of a 3d points using linear search by calculating
        distance to the slicing plane for each point. Points within 0.5 units
        of the slice are classed as belonging to the slice.
        """
        return self.box[x, y, z]


if __name__ == "__main__":
    args = read_args()
    z = ZarrExperiment(args.chunksize, args.ndim, args.points)
    z.find_2d_slice_of_3d_box(args.x, args.y, args.z)
