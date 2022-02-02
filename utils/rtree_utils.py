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


def construct_tree(points: np.ndarray, size: float) -> index.Index:

    props = index.Property(leaf_capacity=1000, dimension=3)
    return index.Index(point_generator(points, size), properties=props)
