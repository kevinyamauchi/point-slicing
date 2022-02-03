from dataclasses import dataclass

import numpy as np
import zarr
from dask import array as da
from scipy.spatial.transform import Rotation as R

from create_data import CreateData
from utils.cli import read_args
from utils.logger import logger
from utils.timer import timer


@dataclass
class LinearSearch:
    data: zarr.Array
    alpha: int
    beta: int
    gamma: int
    x: float
    y: float
    z: float
    tolerance: float

    def __post_init__(self) -> None:
        self.original_points = da.from_zarr(self.data)
        self.plane_point = np.array([self.x, self.y, self.z], dtype=self.data.dtype)
        self.plane_normal = self._create_plane_normal()
        self.projected_points, distance_to_plane = self._project_points_onto_plane()
        self.indices = self._find_points_within_tolerance(distance_to_plane)

    @timer
    def _create_plane_normal(self) -> np.ndarray:
        """
        creates the normal to the plane based on the Euler angles (in degrees)
        * gamma rotation about z-axis
        * beta rotation about y-axis
        * alpha rotation about z-axis
        """
        r = (
            R.from_euler("zyz", [self.gamma, self.beta, self.alpha], degrees=True)
            .as_rotvec()
            .astype(self.data.dtype)
        )
        return r / np.linalg.norm(r)

    @timer
    def _project_points_onto_plane(self) -> tuple[da.Array, da.Array]:
        """
        Project points on to a plane. Plane is defined by a point and a normal
        vector. This function is designed to work with points and planes in 3D.

        Returns
        -------
        projected_point : np.ndarray
            The point that has been projected to the plane.
            This is always an Nx3 array.
        signed_distance_to_plane : np.ndarray
            The signed projection distance between the points and the plane.
            Positive values indicate the point is on the positive normal side
            of the plane.
            Negative values indicate the point is on the negative normal side
            of the plane.
        """
        # get the vector from point on the plane to the point to be projected
        point_vector = self.original_points - self.plane_point

        # find the distance to the plane along the normal direction
        signed_distance_to_plane = point_vector @ self.plane_normal

        # project the point
        projected_points = self.original_points - (
            signed_distance_to_plane.reshape(-1, 1) @ self.plane_normal.reshape(1, -1)
        )

        return projected_points, signed_distance_to_plane

    @timer
    def _find_points_within_tolerance(self, distance: da.Array) -> np.ndarray:
        """
        Find the points within a tolerance of the plane.
        """
        return np.where(np.abs(distance) < self.tolerance)[0]

    @staticmethod
    @timer
    def retrieve_values(data: da.Array, name: str) -> np.ndarray:
        """
        helper function to compute the value of a given dask array
        """
        logger.info(f"Compute {name}")
        return data.compute()


if __name__ == "__main__":
    args = read_args()
    data = CreateData(
        ndim=args.ndim, points_per_dim=args.points, chunk_size=args.chunksize
    ).box
    search = LinearSearch(
        data=data,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        x=args.x,
        y=args.y,
        z=args.z,
        tolerance=args.tolerance,
    )
    idx = search.retrieve_values(search.indices, "indices of points")
    percent = 100 * len(idx) / len(data)
    logger.info(f"found {len(idx)} points within the tolerance, {percent}%")
    found_points = search.retrieve_values(
        search.projected_points[idx], "values of points"
    )
    print(found_points)
