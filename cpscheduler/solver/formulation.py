"""Base class for scheduling problem formulations.

Formulations are responsible for defining the decision variables and build
the model based on the environment's setup, constraints and objective.

The Formulation class provides a registry mechanism for exporting functions that
handle specific types of constraints and objectives. This allows for a modular
and extensible design, where new formulations can be easily created by defining
the necessary variables and registering the appropriate functions for the
constraints and objective of the scheduling problem.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self, final, override

from cpscheduler.environment import (
    Constraint,
    Objective,
    ScheduleSetup,
    SchedulingEnv,
)
from cpscheduler.environment.state import ScheduleState

formulations: dict[str, type["Formulation[Any]"]] = {}

F = TypeVar("F", bound="Formulation[Any]")
E = TypeVar("E", bound=ScheduleSetup | Constraint | Objective)
_T = TypeVar("_T")


Exporter = Callable[[F, ScheduleState, E], None]
VariableExporter = Callable[[F, ScheduleState, E], Any]
Decorator = Callable[[_T], _T]

Register = dict[type[E], Exporter[F, E]]

_Constraint = TypeVar("_Constraint", bound=Constraint)
_Objective = TypeVar("_Objective", bound=Objective)

SolverResult = TypeVar("SolverResult")


class Formulation(Generic[SolverResult], ABC):
    """Base class for scheduling problem formulations."""

    formulation_name: ClassVar[str | None] = None

    _constraint_registry: ClassVar[dict[type[Constraint], Exporter[Any, Any]]]
    _objective_registry: ClassVar[
        dict[type[Objective], VariableExporter[Any, Any]]
    ]

    @classmethod
    def get_constraint_fn(
        cls, constraint: type[Constraint]
    ) -> Exporter[Any, Any]:
        """Get the registered function for a specific constraint type."""
        return cls._constraint_registry[constraint]

    @classmethod
    def get_objective_fn(
        cls, objective: type[Objective]
    ) -> VariableExporter[Any, Any]:
        """Get the registered function for a specific objective type."""
        return cls._objective_registry[objective]

    @override
    def __init_subclass__(cls) -> None:
        cls._constraint_registry = {}
        cls._objective_registry = {}

    @classmethod
    def register_constraint(
        cls: type[F], constraint: type[_Constraint]
    ) -> Decorator[Exporter[F, _Constraint]]:
        """Register a function to handle a specific constraint type.

        This function can be used as a decorator to register the function that
        builds the constraints for a specific constraint type.

        Example
        -------
        >>> @MyFormulation.register_constraint(PrecedenceConstraint)
        ... def precedence_constraint(formulation, state, constraint):
        ...     # Add constraints to the formulation based on the precedence constraint
        ...     pass

        """

        def decorator(fn: Exporter[F, _Constraint]) -> Exporter[F, _Constraint]:
            cls._constraint_registry[constraint] = fn
            return fn

        return decorator

    @classmethod
    def mark_constraint_as_handled(
        cls: type[F], *constraints: type[Constraint]
    ) -> None:
        """Mark a constraint type as handled without adding any constraints to the model.

        This can be used for constraints that are implicitly handled by the formulation
        or do not require explicit constraints in the model.

        Parameters
        ----------
        *constraints: type[Constraint]
            The type of the constraint to mark as handled.

        """
        for constraint in constraints:
            cls._constraint_registry[constraint] = lambda self, state, c: None

    @classmethod
    def register_objective(
        cls: type[F], objective: type[_Objective]
    ) -> Decorator[VariableExporter[F, _Objective]]:
        """Register an objective function for the formulation.

        Differently from setups and constraints, objectives can return a
        variable or an expression representing the objective value.
        This allows for more complex objectives that may require auxiliary
        variables or expressions to be defined in the model.

        It is still the responsibility of the exported function to properly
        set the model's objective to the returned variable or expression.

        Example
        -------
        >>> @MyFormulation.register_objective(MakespanObjective)
        ... def makespan_objective(formulation, state, objective):
        ...     # Define the makespan variable and set it as the objective
        ...     formulation.makespan = ...
        ...     formulation.model.set_objective(formulation.makespan, sense="minimize")
        ...     return formulation.makespan

        """

        def decorator(
            fn: VariableExporter[F, _Objective],
        ) -> VariableExporter[F, _Objective]:
            cls._objective_registry[objective] = fn
            return fn

        return decorator

    @abstractmethod
    def get_assignment(self, task_id: int) -> tuple[int, int]:
        """Get the machine assignment for a specific task.

        Parameters
        ----------
        task_id: int
            The ID of the task to get the assignment for.

        Returns
        -------
        Tuple[int, int]
            A tuple containing the machine ID and the start time assigned to the task.

        """

    @abstractmethod
    def get_objective_value(self) -> float:
        """Get the objective value of the current solution."""

    @abstractmethod
    def initialize_model(self, env: SchedulingEnv) -> None:
        """Initialize the model with variables.

        This should not add any constraints or objective to the model, but only
        initialize the variables based on the environment's setup.
        """

    def post_build(self) -> None:
        """Perform any additional setup after the model has been built.

        This method is called after the model has been built based on the
        environment's setup, constraints and objective.
        It can be used to perform any additional setup or checks that
        require the model to be fully built.
        """

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> SolverResult:
        """Solve the model and return the result.

        The return type can be defined by the specific formulation, but it should
        contain at least the information about whether the solution is optimal or
        not, and any other relevant information about the solving process.

        """

    def warm_start(self, env: SchedulingEnv) -> None:
        """Warm start the formulation with a valid initial schedule."""
        raise NotImplementedError(
            f"warm_start method not available for {type(self).__name__}."
        )

    @final
    def build(
        self: Self, env: SchedulingEnv, symmetry_break: bool = False
    ) -> None:
        """Build the model for the scheduling problem.

        This method initializes the model based on the current instance,
        state, the setup, constraints and objective.
        """
        state = env.state

        for constraint in env.all_constraints:
            if type(constraint) not in self._constraint_registry:
                raise ValueError(
                    f"Constraint type '{type(constraint).__name__}' is not "
                    f"registered for formulation '{type(self).__name__}'."
                )

            self._constraint_registry[type(constraint)](self, state, constraint)

        objective = env.objective
        if type(objective) not in self._objective_registry:
            raise ValueError(
                f"Objective type '{type(objective).__name__}' is not registered "
                f"for formulation '{type(self).__name__}'."
            )

        self._objective_registry[type(objective)](self, state, objective)

        self.post_build()

    def __repr__(self) -> str:
        """Return a generic string representation of the formulation."""
        return f"Formulation(name={self.formulation_name})"


def register_formulation(cls: type[F], name: str) -> type[F]:
    """Register a formulation class with a specific name."""
    if name in formulations:
        raise ValueError(
            f"A formulation with name '{name}' is already registered: "
            f"{formulations[name]}"
        )

    if cls.formulation_name is not None and cls.formulation_name != name:
        raise ValueError(
            f"Formulation class '{cls.__name__}' has a different formulation_name "
            f"('{cls.formulation_name}') than the provided name ('{name}')."
        )

    cls.formulation_name = name
    formulations[name] = cls
    return cls
