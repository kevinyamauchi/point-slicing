from dataclasses import dataclass
from pathlib import Path

import zarr
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
from zarr.errors import PathNotFoundError

from utils.cli import read_args
from utils.timer import timer
from utils.vars import DTYPE, RANDOM_SEED

_file_location = Path(__file__).resolve()


@dataclass
class ZarrExperiment:
    chunk_size: int
    ndim: int
    points_per_dim: int

    def __post_init__(self) -> None:
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
            self.box[:] = self._rng.random(box_shape, dtype=self.box.dtype)

    def extract_slice(self, x: int, y: int, z: int) -> zarr.Array:
        """
        Finds a 2d slice of a 3d points using linear search by calculating
        distance to the slicing plane for each point. Points within 0.5 units
        of the slice are classed as belonging to the slice.
        """
        return R.from_euler("zyx", [z, y, x], degrees=True).as_matrix()


if __name__ == "__main__":
    args = read_args()
    z = ZarrExperiment(args.chunksize, args.ndim, args.points)
    z.extract_slice(args.x, args.y, args.z)
