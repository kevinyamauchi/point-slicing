import pytest

from create_data import CreateData
from utils.vars import DEFAULT_CHUNK_SIZE, DEFAULT_NDIM, DEFAULT_POINTS_PER_DIM


@pytest.fixture(scope="session")
def dummy_points() -> None:
    """
    create some dummy zarr data
    """
    return CreateData(
        ndim=DEFAULT_NDIM,
        points_per_dim=DEFAULT_POINTS_PER_DIM,
        chunk_size=DEFAULT_CHUNK_SIZE,
    ).box
