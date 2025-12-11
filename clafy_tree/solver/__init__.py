"""Solver package (advanced packing)."""

from .advanced.shapes import TreeShape, Placement, EnergyWeights
from .advanced.solver import solve_instance
from .advanced.viz import visualize
from .advanced.energy_sa import energy, simulated_annealing
from .advanced.construction import construct_initial

__all__ = [
    "TreeShape",
    "Placement",
    "EnergyWeights",
    "solve_instance",
    "visualize",
    "energy",
    "simulated_annealing",
    "construct_initial",
]
