"""The single downstream clustering rule used by C0, C1, and C2."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def hierarchy_tree(distance: np.ndarray, *, broad: int, fine: int,
                   method: str = "average") -> dict:
    n = len(distance)
    if method not in {"average", "complete"}:
        raise ValueError("linkage method must be average or complete")
    if distance.shape != (n, n) or not 1 < broad <= fine < n:
        raise ValueError("invalid distance or hierarchy cuts")
    if not np.isfinite(distance).all():
        raise ValueError("distance contains a non-finite value")
    if not np.allclose(distance, distance.T, atol=1e-5):
        raise ValueError("distance is not symmetric")
    if not np.allclose(np.diag(distance), 0):
        raise ValueError("distance diagonal is not zero")
    tree = linkage(squareform(distance, checks=True), method=method)
    return {"linkage": tree,
            "broad": fcluster(tree, broad, criterion="maxclust"),
            "fine": fcluster(tree, fine, criterion="maxclust")}


def average_linkage_tree(distance: np.ndarray, *, broad: int, fine: int) -> dict:
    return hierarchy_tree(distance, broad=broad, fine=fine, method="average")
