from __future__ import annotations

from typing import List

from .geometry import polygon_for_shape
from .shapes import Placement, TreeShape


def visualize(placements: List[Placement], shape: TreeShape, L: float, path: str = "layout.png") -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skip viz")
        return
    polys = [polygon_for_shape(shape, p) for p in placements]
    fig, ax = plt.subplots(figsize=(6, 6))
    # draw square boundary
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "k--", alpha=0.5)
    for poly in polys:
        xs, ys = zip(*poly)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        ax.plot(xs, ys, '-', alpha=0.7)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title(f"n={len(placements)}, L={L:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["visualize"]
