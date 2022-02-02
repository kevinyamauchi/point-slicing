from time import time

import numpy as np
from rtree import index


def point_to_bbox(point: np.ndarray, diameter: float = 0) -> np.ndarray:
    radius = diameter / 2
    mins = point - radius
    maxs = point + radius
    return np.concatenate([mins, maxs], axis=1)


def point_generator(points, size):
    bounding_boxes = point_to_bbox(points, size)

    for i, point in enumerate(bounding_boxes):
        yield (i, point, None)


def construct_tree(points, size, leaf_capacity):
    ndim = points.shape[-1]
    props = index.Property(leaf_capacity=leaf_capacity, dimension=ndim)

    t = time()
    tree = index.Index(point_generator(points, size), properties=props)
    tot = time() - t
    print(
        f"{ndim}D-tree with {len(points)} points and leaf capacity of {leaf_capacity} "
        f"constructed in {tot} seconds"
    )
    return tree


def sweep(tree, n_slices):
    """
    query n consecutive slices in the tree
    """
    mins, maxs = np.reshape(tree.bounds, (-1, tree.properties.dimension))
    min_x = mins[0]
    max_x = maxs[0]
    step_x = (max_x - min_x) / n_slices
    t = time()
    for i in range(n_slices):
        query = (min_x + i * step_x, *mins[1:], min_x + (i + 1) * step_x, *maxs[1:])
        list(tree.intersection(query))
    tot = time() - t
    print(
        f"tree of size {tree.get_size()} swept with {n_slices} slices "
        f"over {tot} seconds ({tot / n_slices} sec/slice)"
    )


points = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])

# blobby points
blobby_points = np.random.normal(size=(1000000, 3), scale=0.2)
centers = np.random.normal(size=(50, 3))
indexes = np.random.normal(
    size=1000000, loc=centers.shape[0] / 2.0, scale=centers.shape[0] / 3.0
)
indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)
scales = 10 ** (np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
blobby_points *= scales
blobby_points += centers[indexes]

# uniform points
random_points = np.random.rand(1000000, 3) * 100


for n_slices in (100, 1000, 10000):
    for leaf_capacity in (10, 100, 1000):
        for typ, pts in (("blobby", blobby_points), ("uniform", random_points)):
            print(f"doing {typ} points: {n_slices=}, {leaf_capacity=}")
            tree = construct_tree(pts, 1, leaf_capacity)
            sweep(tree, n_slices)
