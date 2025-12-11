from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_val(v: str) -> float:
    # values are like "s0.123"
    if v.startswith("s"):
        return float(v[1:])
    return float(v)


def rotate_point(px: float, py: float, theta_deg: float) -> Tuple[float, float]:
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    return px * ct - py * st, px * st + py * ct


DEFAULT_TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
DEFAULT_TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)


def tree_polygon(scale: float) -> List[Tuple[float, float]]:
    xs = DEFAULT_TREE_X * scale
    ys = DEFAULT_TREE_Y * scale
    return list(zip(xs.tolist(), ys.tolist()))


def polygon_at(x: float, y: float, theta: float, scale: float):
    poly = [rotate_point(px, py, theta) for px, py in tree_polygon(scale)]
    return [(x + px, y + py) for px, py in poly]


def plot_submission(sub_path: Path, n: int, scale: float, out_path: Path):
    rows = []
    with sub_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = row["id"].split("_")[0]
            if int(nid) == n:
                rows.append(row)
    if not rows:
        raise ValueError(f"n={n} not found in {sub_path}")

    polys = []
    xs_all: List[float] = []
    ys_all: List[float] = []
    for row in rows:
        x = parse_val(row["x"])
        y = parse_val(row["y"])
        theta = parse_val(row["deg"])
        poly = polygon_at(x, y, theta, scale)
        polys.append(poly)
        for px, py in poly:
            xs_all.append(px)
            ys_all.append(py)

    L = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    for poly in polys:
        xs, ys = zip(*poly)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        ax.plot(xs, ys, "-", alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(min(xs_all) - 0.1, min(xs_all) - 0.1 + L)
    ax.set_ylim(min(ys_all) - 0.1, min(ys_all) - 0.1 + L)
    ax.set_title(f"Layout for n={n} ({len(polys)} trees)")
    ax.plot([min(xs_all) - 0.1, min(xs_all) - 0.1 + L, min(xs_all) - 0.1 + L, min(xs_all) - 0.1, min(xs_all) - 0.1],
            [min(ys_all) - 0.1, min(ys_all) - 0.1, min(ys_all) - 0.1 + L, min(ys_all) - 0.1 + L, min(ys_all) - 0.1],
            "k--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot one n from submission.csv")
    ap.add_argument("--submission", type=Path, default=Path("submission.csv"), help="Path to submission CSV")
    ap.add_argument("--n", type=int, required=True, help="Which n to plot")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale of tree polygon")
    ap.add_argument("--out", type=Path, default=Path("layout_from_submission.png"), help="Output image path")
    args = ap.parse_args()
    plot_submission(args.submission, args.n, args.scale, args.out)


if __name__ == "__main__":
    main()
