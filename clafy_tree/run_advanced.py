from __future__ import annotations

import argparse
import random
from solver.advanced import shapes
from solver.advanced.solver import solve_instance
from solver.advanced.viz import visualize


def main():
    parser = argparse.ArgumentParser(description="Advanced Santa 2025 packer")
    parser.add_argument("n", type=int, help="number of trees (1-200)")
    parser.add_argument("--scale", type=float, default=1.0, help="tree polygon uniform scale")
    parser.add_argument("--restarts", type=int, default=4, help="restarts per L")
    parser.add_argument("--steps", type=int, default=1800, help="SA steps")
    parser.add_argument("--seed", type=int, default=42, help="rng seed")
    parser.add_argument("--viz", action="store_true", help="save layout.png visualization")
    parser.add_argument("--parallel", action="store_true", help="parallelize restarts with processes")
    parser.add_argument("--workers", type=int, default=None, help="max workers for parallel restarts")
    args = parser.parse_args()

    shape = shapes.TreeShape(scale=args.scale)
    rng = random.Random(args.seed)
    L, placements = solve_instance(
        n=args.n,
        shape=shape,
        rng=rng,
        restarts=args.restarts,
        steps_sa=args.steps,
        parallel_restarts=args.parallel,
        max_workers=args.workers,
    )
    print(f"Solved n={args.n}, L~{L:.4f}, placements={len(placements)}")
    for i, p in enumerate(placements):
        print(f"{i}: {p.x:.5f} {p.y:.5f} {p.theta:.3f}")
    if args.viz:
        visualize(placements, shape, L)
        print("saved layout.png")


if __name__ == "__main__":
    main()
