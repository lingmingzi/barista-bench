from __future__ import annotations

import math
from typing import List, Tuple

try:
    import cupy as cp  # type: ignore
    _gpu_available = True
except Exception:
    cp = None  # type: ignore
    _gpu_available = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

from .geometry import SHAPELY_AVAILABLE, polygon_for_shape, polygon_to_shapely
from .shapes import Placement, TreeShape
from .spatial import GridHash


def overlap_area(poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]]) -> float:
    if SHAPELY_AVAILABLE:
        pa = polygon_to_shapely(poly_a)
        pb = polygon_to_shapely(poly_b)
        return pa.intersection(pb).area
    # bounding-circle approximation
    ax, ay = zip(*poly_a)
    bx, by = zip(*poly_b)
    ca = (sum(ax) / len(ax), sum(ay) / len(ay))
    cb = (sum(bx) / len(bx), sum(by) / len(by))
    ra = max(math.hypot(x - ca[0], y - ca[1]) for x, y in poly_a)
    rb = max(math.hypot(x - cb[0], y - cb[1]) for x, y in poly_b)
    d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    if d >= ra + rb:
        return 0.0
    if d <= abs(ra - rb):
        rmin = min(ra, rb)
        return math.pi * rmin * rmin
    phi = math.acos((ra * ra + d * d - rb * rb) / (2 * ra * d)) * 2
    theta = math.acos((rb * rb + d * d - ra * ra) / (2 * rb * d)) * 2
    return 0.5 * (ra * ra * (phi - math.sin(phi)) + rb * rb * (theta - math.sin(theta)))


def polygons_overlap(poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]], shape: TreeShape) -> bool:
    if SHAPELY_AVAILABLE:
        pa = polygon_to_shapely(poly_a)
        pb = polygon_to_shapely(poly_b)
        return pa.intersects(pb)
    ax, ay = zip(*poly_a)
    bx, by = zip(*poly_b)
    ca = (sum(ax) / len(ax), sum(ay) / len(ay))
    cb = (sum(bx) / len(bx), sum(by) / len(by))
    ra = shape.bounding_radius
    rb = shape.bounding_radius
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < ra + rb - 1e-9


def in_bounds(poly: List[Tuple[float, float]], L: float) -> bool:
    return all(0.0 <= x <= L and 0.0 <= y <= L for x, y in poly)


def is_feasible(placements: List[Placement], shape: TreeShape, L: float) -> bool:
    polys = [polygon_for_shape(shape, p) for p in placements]

    # Bounds check (vectorized if numpy available)
    if np is not None:
        all_pts = np.array([pt for poly in polys for pt in poly], dtype=float)
        xs = all_pts[:, 0]
        ys = all_pts[:, 1]
        if (xs < 0).any() or (xs > L).any() or (ys < 0).any() or (ys > L).any():
            return False
    else:
        if not all(in_bounds(poly, L) for poly in polys):
            return False

    # Overlap check
    if SHAPELY_AVAILABLE:
        pa = [polygon_to_shapely(poly) for poly in polys]
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                if pa[i].intersects(pa[j]):
                    return False
        return True

    # Fast radius overlap using GPU (cupy) then numpy
    rad = shape.bounding_radius
    if _gpu_available and cp is not None:
        centers = cp.asarray([[p.x, p.y] for p in placements], dtype=cp.float32)
        diff = centers[:, None, :] - centers[None, :, :]
        dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
        mask = cp.triu(dist2 < (2 * rad) ** 2 - 1e-9, k=1)
        if mask.any():
            return False
        return True
    if np is not None:
        centers = np.array([[p.x, p.y] for p in placements], dtype=float)
        diff = centers[:, None, :] - centers[None, :, :]
        dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
        mask = np.triu(dist2 < (2 * rad) ** 2 - 1e-9, k=1)
        return not mask.any()

    # Fallback grid + polygon overlap
    cell = shape.bounding_radius * 2
    grid = GridHash(cell)
    for idx, p in enumerate(placements):
        grid.insert(idx, p.x, p.y)
    for i, p in enumerate(placements):
        for j in grid.nearby(p.x, p.y):
            if j <= i:
                continue
            if polygons_overlap(polys[i], polys[j], shape):
                return False
    return True


__all__ = ["overlap_area", "polygons_overlap", "in_bounds", "is_feasible"]
