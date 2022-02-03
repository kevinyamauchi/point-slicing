import pytest
import zarr


@pytest.fixture(scope="session")
def point_in_plane() -> zarr.Array:
    """
    create a point in the default plane
    """
    z = zarr.empty(shape=(1, 3))
    z[:] = [[0.3001, 0.1001, 0.5001]]
    return z


@pytest.fixture(scope="session")
def point_not_in_plane() -> zarr.Array:
    """
    create a point outside of the default plane
    """
    z = zarr.empty(shape=(1, 3))
    z[:] = [[0.1, 0.1, 0.1]]
    return z
