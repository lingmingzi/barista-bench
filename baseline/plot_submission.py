import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tree polygon template (same as geometry.py)
TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                   -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2,
                   -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)


def parse_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['n'] = df['id'].str.slice(0, 3).astype(int)
    df['idx'] = df['id'].str.split('_').str[1].astype(int)
    df['x'] = df['x'].astype(str).str.lstrip('s').astype(float)
    df['y'] = df['y'].astype(str).str.lstrip('s').astype(float)
    df['deg'] = df['deg'].astype(str).str.lstrip('s').astype(float)
    return df


def rotate_poly(cx: float, cy: float, deg: float) -> np.ndarray:
    rad = deg * math.pi / 180.0
    c = math.cos(rad)
    s = math.sin(rad)
    px = TREE_X * c - TREE_Y * s + cx
    py = TREE_X * s + TREE_Y * c + cy
    return np.stack([px, py], axis=1)


def plot_n(df: pd.DataFrame, n: int, out_dir: Path, show: bool):
    sub = df[df['n'] == n].sort_values('idx')
    if sub.empty:
        print(f"Skip N={n}: not found in submission")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in sub.iterrows():
        poly = rotate_poly(row['x'], row['y'], row['deg'])
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.5, edgecolor='k', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'N={n} ({len(sub)} trees)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # expand limits a bit for better view
    all_x = sub['x'].to_numpy()
    all_y = sub['y'].to_numpy()
    pad = 1.0
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    out_path = out_dir / f'plot_n{n:03d}.png'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved {out_path}')


def parse_n_list(n_arg: Optional[List[int]], df: pd.DataFrame) -> List[int]:
    if n_arg:
        return n_arg
    return sorted(df['n'].unique())


def main():
    parser = argparse.ArgumentParser(description='Plot tree layout from submission CSV')
    parser.add_argument('--input', type=str, required=True, help='Submission CSV path')
    parser.add_argument('--n', type=int, nargs='*', default=None, help='List of N to plot (default: all)')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Also show interactively')
    args = parser.parse_args()

    csv_path = Path(args.input)
    out_dir = Path(args.output_dir)

    df = parse_submission(csv_path)
    n_list = parse_n_list(args.n, df)
    print(f'Plotting N values: {n_list}')
    for n in n_list:
        plot_n(df, n, out_dir, show=args.show)


if __name__ == '__main__':
    main()
