"""Module for scheduling problem formulations and solvers."""

__all__ = [
    "DisjunctiveMILPFormulation",
    "Formulation",
    "SchedulingSolver",
    # "DisjunctiveCPFormulation",
]

from .formulation import Formulation, formulations, register_formulation
from .milp.disjunctive.formulation import DisjunctiveMILPFormulation
from .solver import SchedulingSolver

# from .cp import DisjunctiveCPFormulation

register_formulation(DisjunctiveMILPFormulation, "disjunctive")
# register_formulation(DisjunctiveCPFormulation, "disjunctive_cp")


def get_formulations() -> list[str]:
    return list(formulations.keys())
