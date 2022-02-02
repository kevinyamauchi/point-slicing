from dataclasses import dataclass

import numpy as np
import zarr

from create_data import CreateData
from utils.cli import read_args
from utils.linear_utils import (
    create_plane_normal,
    find_points_within_tolerance,
    project_points_onto_plane,
)


@dataclass
class LinearSearch:
    box: zarr.Array
    alpha: int
    beta: int
    gamma: int
    x: float
    y: float
    z: float
    tolerance: float

    def __post_init__(self) -> None:
        self.plane_point = np.array([self.x, self.y, self.z])
        self.plane_normal = create_plane_normal(self.alpha, self.beta, self.gamma)
        self.points, distance_to_plane = project_points_onto_plane(
            self.box, self.plane_point, self.plane_normal
        )
        self.indices = find_points_within_tolerance(distance_to_plane, self.tolerance)


if __name__ == "__main__":
    args = read_args()
    box = CreateData(
        ndim=args.ndim, points_per_dim=args.points, chunk_size=args.chunksize
    ).box
    z = LinearSearch(
        box=box,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        x=args.x,
        y=args.y,
        z=args.z,
        tolerance=args.tolerance,
    )
    print(z.points[z.indices])
