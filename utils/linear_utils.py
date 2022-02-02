import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger import logger


def create_plane_normal(alpha: int, beta: int, gamma: int) -> np.ndarray:
    """
    creates the normal to the plane based on the Euler angles (in degrees)
    * gamma rotation about z-axis
    * beta rotation about y-axis
    * alpha rotation about z-axis
    """
    r = R.from_euler("zyz", [gamma, beta, alpha], degrees=True)
    return r.as_rotvec()


def project_points_onto_plane(
    points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project points on to a plane. Plane is defined by a point and a normal
    vector. This function is designed to work with points and planes in 3D.

    Parameters
    ----------
    points : np.ndarray
        The coordinate of the point to be projected. The points
        should be 3D and have shape shape (N,3) for N points.
    plane_point : np.ndarray
        The point on the plane used to define the plane.
        Should have shape (3,).
    plane_normal : np.ndarray
        The normal vector used to define the plane.
        Should be a unit vector and have shape (3,).

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
    # make sure both points and plane point are arrays
    points = np.atleast_2d(points)
    plane_point = np.asarray(plane_point)

    # make the plane normals have the same shape as the points
    plane_normal = np.tile(plane_normal, (points.shape[0], 1))

    # get the vector from point on the plane to the point to be projected
    point_vector = points - plane_point

    # find the distance to the plane along the normal direction
    signed_distance_to_plane = np.multiply(point_vector, plane_normal).sum(axis=1)

    # project the point
    projected_points = points - (signed_distance_to_plane[:, np.newaxis] * plane_normal)

    return projected_points, signed_distance_to_plane


def find_points_within_tolerance(distance: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Find the points within a tolerance of the plane.
    """
    indices = np.argwhere(np.abs(distance) < tolerance)[:, 0]
    logger.info(f"found {len(indices)} points within the tolerance")
    return indices
