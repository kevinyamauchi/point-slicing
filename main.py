from timeit import default_timer as timer

import numpy as np
import zarr
from numpy.random import default_rng

DTYPE = np.float32
RANDOM_SEED = 30


def create_zarr_random_points(
    dims: int, points_per_dim: int, chunk_size: int = 1_000
) -> zarr.Array:
    """
    creates a d-dimensional zarr dataset of radnomly distributed points
    """
    # initialise empty zarr array
    box = zarr.zeros((points_per_dim,) * dims, chunks=(chunk_size,) * dims, dtype=DTYPE)

    # initialise random number generator
    rng = default_rng(RANDOM_SEED)

    # fill zarr array with random points from uniform distribution
    box[:] = rng.random(size=box.shape, dtype=DTYPE)
    return box


if __name__ == "__main__":
    start = timer()
    z = create_zarr_random_points(3, 10_000)
    end = timer()
    print(f"Time taken: {end - start}")
