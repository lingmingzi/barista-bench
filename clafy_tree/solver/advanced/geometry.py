from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

try:
    import shapely  # type: ignore
    import shapely.geometry  # type: ignore

    SHAPELY_AVAILABLE = True
except Exception:
    shapely = None  # type: ignore
    SHAPELY_AVAILABLE = False

from .shapes import Placement, TreeShape


def rotate_point(px: float, py: float, theta_deg: float) -> Tuple[float, float]:
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    return px * ct - py * st, px * st + py * ct


def polygon_for_shape(shape: TreeShape, placement: Placement):
    poly = [rotate_point(px, py, placement.theta) for px, py in shape.polygon()]
    return [(placement.x + px, placement.y + py) for px, py in poly]


def polygon_to_shapely(poly: Sequence[Tuple[float, float]]):
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("Shapely not available")
    return shapely.geometry.Polygon(poly)


def bbox_of_points(points: Iterable[Tuple[float, float]]):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


__all__ = [
    "SHAPELY_AVAILABLE",
    "rotate_point",
    "polygon_for_shape",
    "polygon_to_shapely",
    "bbox_of_points",
]
