from linear_search import LinearSearch
from utils.vars import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_TOL,
    DEFAULT_X,
    DEFAULT_Y,
    DEFAULT_Z,
)


def test_point_in_plane(point_in_plane) -> None:
    """
    checks a test point is in the plane
    """
    search = LinearSearch(
        data=point_in_plane,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        gamma=DEFAULT_GAMMA,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        z=DEFAULT_Z,
        tolerance=DEFAULT_TOL,
    )
    indices = search.retrieve_values(search.indices, "indices of points")
    assert len(indices) >= 1, "point should exist in plane but doesn't"


def test_point_not_in_plane(point_not_in_plane) -> None:
    """
    checks a test point is outside of the plane
    """
    search = LinearSearch(
        data=point_not_in_plane,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        gamma=DEFAULT_GAMMA,
        x=DEFAULT_X,
        y=DEFAULT_Y,
        z=DEFAULT_Z,
        tolerance=DEFAULT_TOL,
    )
    indices = search.retrieve_values(search.indices, "indices of points")
    assert len(indices) == 0, "point exists in plane but shouldn't"
