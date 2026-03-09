"""
symmetry_breaking.py

This module contains some implementations of symmetry breaking constraints. 
Symmetry breaking constraints are used to reduce the search space of the solver
by eliminating symmetric solutions.
For example, in parallel machine scheduling, if two machines are identical, we
can add a constraint that forces the first task to be assigned to the first
machine, which breaks the initial symmetry between the machines.

MILP formulation are often affected by symmetries in the problem, so this
module is particularly relevant for those formulations, exposing a common
interface.
"""
