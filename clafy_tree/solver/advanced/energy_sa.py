from __future__ import annotations

import math
import random
from typing import List

try:
    import cupy as cp  # type: ignore
    _gpu_available = True
except Exception:
    cp = None  # type: ignore
    _gpu_available = False

try:
    import numpy as np  # type: ignore
except Exception:  # numpy may be unavailable in some Kaggle runtimes
    np = None  # type: ignore

from .collision import polygons_overlap
from .geometry import SHAPELY_AVAILABLE, polygon_for_shape, polygon_to_shapely
from .shapes import EnergyWeights, Placement, TreeShape
from .spatial import GridHash


def energy(placements: List[Placement], shape: TreeShape, L: float, weights: EnergyWeights) -> float:
    polys = [polygon_for_shape(shape, p) for p in placements]
    e_bbox = L
    e_overlap = 0.0
    e_out = 0.0
    e_disp = 0.0
    count = len(placements)

    if SHAPELY_AVAILABLE:
        import shapely.geometry  # type: ignore

        shps = [polygon_to_shapely(poly) for poly in polys]
        box = shapely.geometry.box(0, 0, L, L)
        for i in range(count):
            if not shps[i].within(box):
                e_out += shps[i].difference(box).area
            for j in range(i + 1, count):
                inter = shps[i].intersection(shps[j])
                if not inter.is_empty:
                    e_overlap += inter.area
    else:
        for poly in polys:
            for x, y in poly:
                if x < 0 or x > L or y < 0 or y > L:
                    e_out += 1.0
        grid = GridHash(shape.bounding_radius * 2)
        for i, p in enumerate(placements):
            grid.insert(i, p.x, p.y)
        for i, p in enumerate(placements):
            for j in grid.nearby(p.x, p.y):
                if j <= i:
                    continue
                if polygons_overlap(polys[i], polys[j], shape):
                    e_overlap += 1.0
    if count > 1:
        if _gpu_available and cp is not None:
            coords = cp.asarray([[p.x, p.y] for p in placements], dtype=cp.float32)
            diff = coords[:, None, :] - coords[None, :, :]
            dist = cp.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
            triu = cp.triu_indices(count, k=1)
            pair_d = dist[triu]
            if pair_d.size:
                e_disp = float(cp.mean(pair_d).get())
        elif np is not None:
            coords = np.array([[p.x, p.y] for p in placements], dtype=float)
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            triu = np.triu_indices(count, k=1)
            pair_d = dist[triu]
            if pair_d.size:
                e_disp = float(pair_d.mean())
        else:
            total_d = 0.0
            pairs = 0
            for i in range(count):
                for j in range(i + 1, count):
                    total_d += math.hypot(placements[i].x - placements[j].x, placements[i].y - placements[j].y)
                    pairs += 1
            e_disp = total_d / pairs
    return (
        weights.alpha * e_bbox
        + weights.beta * e_overlap
        + weights.gamma * e_out
        - weights.delta * e_disp
    )


def simulated_annealing(
    placements: List[Placement],
    shape: TreeShape,
    L: float,
    weights: EnergyWeights,
    steps: int,
    temp0: float,
    temp1: float,
    move_scale: float,
    rng: random.Random,
) -> List[Placement]:
    current = placements
    best = placements
    e_best = energy(best, shape, L, weights)
    e_cur = e_best
    for step in range(steps):
        t = temp0 * (temp1 / temp0) ** (step / max(1, steps - 1))
        move_frac = move_scale * max(0.05, (steps - step) / steps)
        cand = current.copy()
        ops = ["translate", "rotate"]
        if len(cand) > 1:
            ops.append("swap")
        op = rng.choice(ops)
        if op == "translate":
            i = rng.randrange(len(cand))
            dx = (rng.random() * 2 - 1) * move_frac * L
            dy = (rng.random() * 2 - 1) * move_frac * L
            cand[i] = Placement(
                x=min(max(0.0, cand[i].x + dx), L),
                y=min(max(0.0, cand[i].y + dy), L),
                theta=cand[i].theta,
            )
        elif op == "rotate":
            i = rng.randrange(len(cand))
            dtheta = (rng.random() * 2 - 1) * 15.0
            cand[i] = Placement(cand[i].x, cand[i].y, (cand[i].theta + dtheta) % 360)
        else:  # swap
            i, j = rng.sample(range(len(cand)), 2)
            cand[i], cand[j] = cand[j], cand[i]

        e_new = energy(cand, shape, L, weights)
        delta = e_new - e_cur
        if delta < 0 or rng.random() < math.exp(-delta / max(1e-9, t)):
            current = cand
            e_cur = e_new
            if e_new < e_best:
                best = cand
                e_best = e_new
    return best


__all__ = ["energy", "simulated_annealing"]
