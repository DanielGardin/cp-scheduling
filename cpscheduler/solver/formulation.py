from typing import Any, TypeVar, ClassVar
from typing_extensions import Self
from collections.abc import Callable
from abc import ABC, abstractmethod

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment import (
    SchedulingEnv,
    ScheduleSetup,
    Constraint,
    Objective,
)

formulations: dict[str, type["Formulation"]] = {}

F = TypeVar("F", bound="Formulation")
E = TypeVar("E", bound=ScheduleSetup | Constraint | Objective)
_T = TypeVar("_T")


Exporter = Callable[[F, ScheduleState, E], None]
VariableExporter = Callable[[F, ScheduleState, E], Any]
Decorator = Callable[[_T], _T]

Register = dict[type[E], Exporter[F, E]]

_Constraint = TypeVar("_Constraint", bound=Constraint)
_Objective = TypeVar("_Objective", bound=Objective)


class Formulation(ABC):
    formulation_name: ClassVar[str | None] = None

    _constraint_registry: ClassVar[dict[type[Constraint], Exporter[Any, Any]]]
    _objective_registry: ClassVar[
        dict[type[Objective], VariableExporter[Any, Any]]
    ]

    def __init_subclass__(cls) -> None:
        cls._constraint_registry = {}
        cls._objective_registry = {}

    @classmethod
    def register_constraint(
        cls: type[F], constraint: type[_Constraint]
    ) -> Decorator[Exporter[F, _Constraint]]:
        def decorator(fn: Exporter[F, _Constraint]) -> Exporter[F, _Constraint]:
            cls._constraint_registry[constraint] = fn
            return fn

        return decorator

    @classmethod
    def mark_constraint_as_handled(
        cls: type[F], *constraints: type[Constraint]
    ) -> None:
        """
        Mark a constraint type as handled without adding any constraints to the model.
        This can be used for constraints that are implicitly handled by the formulation
        or do not require explicit constraints in the model.
        Args:
            constraint: type[_Constraint]
                The type of the constraint to mark as handled.
        """
        for constraint in constraints:
            cls._constraint_registry[constraint] = lambda self, state, c: None

    @classmethod
    def register_objective(
        cls: type[F], objective: type[_Objective]
    ) -> Decorator[VariableExporter[F, _Objective]]:
        """
        Register an objective function for the formulation.

        Differently from setups and constraints, objectives can return a
        variable or an expression representing the objective value.
        This allows for more complex objectives that may require auxiliary
        variables or expressions to be defined in the model.

        It is still the responsibility of the exported function to properly
        set the model's objective to the returned variable or expression.

        Usage example:
        ```python
        @Formulation.register_objective(Makespan)
        def makespan_objective(formulation, state, objective):
            <define variables and constraints for makespan objective>
            return makespan_variable
        ```
        """

        def decorator(
            fn: VariableExporter[F, _Objective],
        ) -> VariableExporter[F, _Objective]:
            cls._objective_registry[objective] = fn
            return fn

        return decorator

    @abstractmethod
    def get_assignment(self, task_id: int) -> tuple[int, int]:
        """
        Get the machine assignment for a specific task.
        Args:
            task_id: int
                The ID of the task to get the assignment for.
        Returns:
            A tuple (start_time, machine_id) representing the assignment.
        """

    @abstractmethod
    def get_objective_value(self) -> float:
        """
        Get the objective value of the current solution.
        Returns:
            The objective value as a float.
        """

    @abstractmethod
    def initialize_model(self, env: SchedulingEnv) -> None:
        """
        Initialize the model with variables.

        This should not add any constraints or objective to the model, but only
        initialize the variables based on the environment's setup.
        """

    def finalize_model(self, env: SchedulingEnv) -> None:
        """
        Finalize the model after all constraints and objective have been added.

        This can be used to add any necessary constraints or modifications to the
        model that depend on the complete set of constraints and objective being
        defined.
        """

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """
        Solve the model using the specified solver configuration.
        Args:
            solver_config: dict
                A dictionary containing solver-specific configuration parameters.
        Returns:
            The result of the solve operation, which may include status, objective value, etc.
        """

    def warm_start(self, env: SchedulingEnv) -> None:
        raise NotImplementedError(
            f"warm_start method not available for {type(self).__name__}."
        )

    def build(
        self: Self, env: SchedulingEnv, symmetry_break: bool = False
    ) -> None:
        """
        Build the model for the scheduling problem.

        This method initializes the model based on the current instance,
        state, the setup, constraints and objective.
        """
        state = env.state

        for constraint in env.setup_constraints + env.constraints:
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

    def __repr__(self) -> str:
        return f"Formulation(name={self.formulation_name})"


def register_formulation(cls: type[F], name: str) -> type[F]:
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
