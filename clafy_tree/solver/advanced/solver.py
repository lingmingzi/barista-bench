from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from .collision import is_feasible
from .construction import construct_initial
from .energy_sa import simulated_annealing
from .shapes import EnergyWeights, Placement, TreeShape


def solve_instance(
    n: int,
    shape: TreeShape,
    rng: Optional[random.Random] = None,
    weights: Optional[EnergyWeights] = None,
    restarts: int = 3,
    steps_sa: int = 1500,
    temp0: float = 0.1,
    temp1: float = 0.001,
    move_scale: float = 0.15,
    eps: float = 1e-3,
    parallel_restarts: bool = False,
    max_workers: Optional[int] = None,
) -> Tuple[float, List[Placement]]:
    rng = rng or random.Random(42)
    weights = weights or EnergyWeights()

    area = shape.area
    L_low = math.sqrt(n * area) * 0.6
    L_high = math.sqrt(n * area) * 4.0

    best_sol: List[Placement] = []
    best_L = L_high

    while L_high - L_low > eps:
        L_mid = 0.5 * (L_low + L_high)
        feasible = False
        candidate_sol: List[Placement] = []

        if parallel_restarts and restarts > 1:
            seeds = [rng.getrandbits(32) for _ in range(restarts)]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _single_restart,
                        n,
                        L_mid,
                        shape,
                        weights,
                        steps_sa,
                        temp0,
                        temp1,
                        move_scale,
                        seed,
                    ): seed
                    for seed in seeds
                }
                for fut in as_completed(futures):
                    opt = fut.result()
                    if opt is not None:
                        feasible = True
                        candidate_sol = opt
                        break
        else:
            for _ in range(restarts):
                init = construct_initial(n, L_mid, shape, rng)
                opt = simulated_annealing(init, shape, L_mid, weights, steps_sa, temp0, temp1, move_scale, rng)
                if is_feasible(opt, shape, L_mid):
                    feasible = True
                    candidate_sol = opt
                    break
        if feasible:
            best_sol = candidate_sol
            best_L = L_mid
            L_high = L_mid
        else:
            L_low = L_mid
    return best_L, best_sol


__all__ = ["solve_instance"]


def _single_restart(
    n: int,
    L: float,
    shape: TreeShape,
    weights: EnergyWeights,
    steps_sa: int,
    temp0: float,
    temp1: float,
    move_scale: float,
    seed: int,
) -> Optional[List[Placement]]:
    rng = random.Random(seed)
    init = construct_initial(n, L, shape, rng)
    opt = simulated_annealing(init, shape, L, weights, steps_sa, temp0, temp1, move_scale, rng)
    if is_feasible(opt, shape, L):
        return opt
    return None
