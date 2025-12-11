from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
from typing import Set

from solver import TreeShape, solve_instance, visualize
from solver.advanced.collision import is_feasible


def format_val(v: float) -> str:
    return f"s{v:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission with advanced solver (binary search + SA)")
    parser.add_argument("--start-n", type=int, default=1, help="Start n (inclusive)")
    parser.add_argument("--end-n", type=int, default=200, help="End n (inclusive)")
    parser.add_argument("--scale", type=float, default=1.0, help="Tree polygon uniform scale")
    parser.add_argument("--restarts", type=int, default=6, help="Restarts per L")
    parser.add_argument("--steps", type=int, default=2000, help="SA steps")
    parser.add_argument("--temp0", type=float, default=0.1, help="Initial temperature")
    parser.add_argument("--temp1", type=float, default=0.001, help="Final temperature")
    parser.add_argument("--move-scale", type=float, default=0.15, help="Base move scale fraction of L")
    parser.add_argument("--seed", type=int, default=123, help="Global RNG seed")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Output CSV path")
    parser.add_argument("--parallel", action="store_true", help="Parallelize restarts with processes")
    parser.add_argument("--workers", type=int, default=None, help="Max workers for parallel restarts")
    parser.add_argument("--plot-n", type=int, default=None, help="If set, visualize this n")
    parser.add_argument("--plot-path", type=Path, default=Path("layout.png"), help="Path for visualization output")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output; skip completed n")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shape = TreeShape(scale=args.scale)

    rows = []
    captured = None

    completed: Set[int] = set()
    if args.resume and args.output.exists():
        with args.output.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                n_id = int(row["id"].split("_")[0])
                completed.add(n_id)
    # prepare writer in append mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or not args.resume
    f_out = args.output.open("a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=["id", "x", "y", "deg"])
    if write_header:
        writer.writeheader()

    for n in range(args.start_n, args.end_n + 1):
        if n in completed:
            print(f"skip n={n:03d} (already in output)")
            continue
        L, placements = solve_instance(
            n=n,
            shape=shape,
            rng=rng,
            restarts=args.restarts,
            steps_sa=args.steps,
            temp0=args.temp0,
            temp1=args.temp1,
            move_scale=args.move_scale,
            parallel_restarts=args.parallel,
            max_workers=args.workers,
        )
        if not is_feasible(placements, shape, L):
            raise RuntimeError(f"Feasibility check failed at n={n}")
        if args.plot_n == n:
            captured = (L, placements)
        for i, p in enumerate(placements):
            row = {
                "id": f"{n:03d}_{i}",
                "x": format_val(p.x),
                "y": format_val(p.y),
                "deg": format_val(p.theta),
            }
            writer.writerow(row)
        f_out.flush()
        print(f"n={n:03d} L={L:.4f} placed={len(placements)} -> appended")

    f_out.close()
    print(f"Wrote/updated submission at {args.output}")

    if captured is not None:
        Lcap, placap = captured
        visualize(placap, shape, Lcap, path=str(args.plot_path))
        print(f"Saved visualization for n={args.plot_n} to {args.plot_path}")


if __name__ == "__main__":
    main()
