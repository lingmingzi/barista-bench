from __future__ import annotations

import math
import random
from typing import List, Tuple

from .shapes import Placement, TreeShape


def hex_layout(n: int, L: float) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    if n == 1:
        return [(L / 2, L / 2)]
    spacing = L / math.sqrt(n)
    h = spacing * math.sqrt(3) / 2
    rows = int(L // h) + 2
    cols = int(L // spacing) + 2
    for r in range(rows):
        offset = 0.5 * spacing if r % 2 else 0.0
        y = r * h
        if y > L:
            break
        for c in range(cols):
            x = offset + c * spacing
            if x > L:
                break
            coords.append((x, y))
    return coords[:n]


def spiral_layout(n: int, L: float) -> List[Tuple[float, float]]:
    pts = []
    cx = cy = L / 2
    r = L * 0.45
    turns = 3
    for i in range(n):
        t = 2 * math.pi * turns * (i / max(1, n - 1))
        rr = r * (i / max(1, n - 1))
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    return pts


def grid_layout(n: int, L: float) -> List[Tuple[float, float]]:
    side = math.ceil(math.sqrt(n))
    spacing = L / (side + 1)
    pts = []
    for r in range(side):
        for c in range(side):
            if len(pts) >= n:
                break
            pts.append(((c + 1) * spacing, (r + 1) * spacing))
    return pts


def detect_hexish(points: List[Tuple[float, float]], tol: float = 0.1) -> bool:
    if len(points) < 4:
        return False
    dists = []
    for i in range(min(len(points), 20)):
        for j in range(i + 1, min(len(points), 20)):
            dists.append(math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]))
    if not dists:
        return False
    avg = sum(dists) / len(dists)
    var = sum((d - avg) ** 2 for d in dists) / len(dists)
    return math.sqrt(var) / (avg + 1e-9) < tol


def align_to_grid(points: List[Tuple[float, float]], L: float) -> List[Tuple[float, float]]:
    if not points:
        return points
    xs, ys = zip(*points)
    step = max(min(L, (max(xs) - min(xs)) / max(1, len(set(xs)))), L * 0.01)
    snap = lambda v: round(v / step) * step  # noqa: E731
    return [(snap(x), snap(y)) for x, y in points]


def construct_initial(n: int, L: float, shape: TreeShape, rng: random.Random) -> List[Placement]:
    if n <= 10:
        coords = grid_layout(n, L)
    elif n <= 50:
        coords = hex_layout(n, L)
    else:
        coords = hex_layout(n, L)
    if detect_hexish(coords):
        coords = align_to_grid(coords, L)
    placements = [Placement(x, y, rng.uniform(0, 180)) for x, y in coords]
    return placements


__all__ = [
    "hex_layout",
    "spiral_layout",
    "grid_layout",
    "detect_hexish",
    "align_to_grid",
    "construct_initial",
]
