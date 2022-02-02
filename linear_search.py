from dataclasses import dataclass

import numpy as np

from create_data import CreateData
from utils.cli import read_args
from utils.linear_utils import (
    create_plane_normal,
    find_points_within_tolerance,
    project_points_onto_plane,
)


@dataclass
class LinearSearch(CreateData):
    alpha: int
    beta: int
    gamma: int
    x: float
    y: float
    z: float
    tolerance: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.plane_point = np.array([self.x, self.y, self.z])
        self.plane_normal = create_plane_normal(self.alpha, self.beta, self.gamma)
        self.points, distance_to_plane = project_points_onto_plane(
            self.box, self.plane_point, self.plane_normal
        )
        self.indices = find_points_within_tolerance(distance_to_plane, self.tolerance)


if __name__ == "__main__":
    args = read_args()
    z = LinearSearch(
        args.chunksize,
        args.ndim,
        args.points,
        args.alpha,
        args.beta,
        args.gamma,
        args.x,
        args.y,
        args.z,
        args.tolerance,
    )
    print(z.points[z.indices])
