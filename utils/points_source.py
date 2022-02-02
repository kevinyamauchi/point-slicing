from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PointsSource(ABC):
    def __init__(self, data):
        self._data = data

    @abstractmethod
    def slice_data(self, slice_indices):
        raise NotImplementedError

    @abstractmethod
    def ray_intersection(
        self, ray_start: np.ndarray, ray_direction: np.ndarray
    ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def nearest_neighbor(self, coordinate: np.ndarray) -> int:
        raise NotImplementedError
