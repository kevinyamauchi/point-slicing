from typing import List

import dask.array as da
import numpy as np

from .points_source import PointsSource
from .rtree_utils import construct_tree

POINT_SIZE = 1


class PointsRTree(PointsSource):
    def __init__(self, file_path: str):
        self._file_path = file_path
        dask_array = self._load_data(file_path)

        super().__init__(dask_array)

        print(self._data_array)

        self._initialize_tree(self._data)

    def _load_data(self, file_path):
        return da.from_zarr(file_path)

    def _initialize_tree(self, zarr_array):
        self._tree = construct_tree(zarr_array, POINT_SIZE)

    def slice_data(
        self, slice_point: np.ndarray, slice_normal: np.ndarray, slice_thickness
    ):
        pass

    def ray_intersection(
        self, ray_start: np.ndarray, ray_direction: np.ndarray
    ) -> List[int]:
        raise NotImplementedError

    def nearest_neighbor(self, coordinate: np.ndarray) -> int:
        raise NotImplementedError
