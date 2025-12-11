from __future__ import annotations

import math
from typing import Dict, List, Tuple


class GridHash:
    def __init__(self, cell: float):
        self.cell = cell
        self.cells: Dict[Tuple[int, int], List[int]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def insert(self, idx: int, x: float, y: float) -> None:
        key = self._key(x, y)
        self.cells.setdefault(key, []).append(idx)

    def nearby(self, x: float, y: float) -> List[int]:
        gx, gy = self._key(x, y)
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.extend(self.cells.get((gx + dx, gy + dy), []))
        return out


__all__ = ["GridHash"]
