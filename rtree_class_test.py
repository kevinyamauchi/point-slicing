from utils.points_rtree import PointsRTree

rtree_source = PointsRTree("./data/random_points_dim3_points10.zarr")

print(rtree_source._tree)
