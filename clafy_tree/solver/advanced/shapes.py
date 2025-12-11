from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


# Default tree polygon provided by user
DEFAULT_TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
DEFAULT_TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)


@dataclass
class TreeShape:
    scale: float = 1.0  # uniform scale of the provided polygon

    def polygon(self) -> List[Tuple[float, float]]:
        xs = DEFAULT_TREE_X * self.scale
        ys = DEFAULT_TREE_Y * self.scale
        return list(zip(xs.tolist(), ys.tolist()))

    @property
    def area(self) -> float:
        xs = DEFAULT_TREE_X * self.scale
        ys = DEFAULT_TREE_Y * self.scale
        return float(0.5 * np.abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))

    @property
    def bounding_radius(self) -> float:
        xs = DEFAULT_TREE_X * self.scale
        ys = DEFAULT_TREE_Y * self.scale
        return float(np.max(np.hypot(xs, ys)))


@dataclass
class Placement:
    x: float
    y: float
    theta: float  # degrees


@dataclass
class EnergyWeights:
    alpha: float = 1.0  # bbox side
    beta: float = 50.0  # overlap area
    gamma: float = 50.0  # out-of-bounds area
    delta: float = 0.1  # dispersion (negative encourages tightness)


__all__ = ["TreeShape", "Placement", "EnergyWeights"]
