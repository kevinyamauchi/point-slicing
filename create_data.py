from dataclasses import dataclass
from pathlib import Path

import zarr
from numpy.random import default_rng
from zarr.errors import PathNotFoundError

from utils.timer import timer
from utils.vars import DTYPE, RANDOM_SEED

_file_location = Path(__file__).resolve()


@dataclass
class CreateData:
    ndim: int
    points_per_dim: int
    chunk_size: int

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
        try:
            self.box = zarr.open(self._data_path, mode="r")
        except PathNotFoundError:
            self.box = zarr.open(
                self._data_path,
                mode="w",
                shape=(self.points_per_dim, self.ndim),
                chunks=(self.chunk_size, self.ndim),
                dtype=DTYPE,
            )
            self.box[:] = self._rng.random(self.box.shape, dtype=self.box.dtype)
