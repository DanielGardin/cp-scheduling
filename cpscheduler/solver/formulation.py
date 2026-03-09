from typing import Any, TypeVar, ClassVar, Generic, get_args, get_origin
from typing_extensions import Self, get_original_bases
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
_Setup = TypeVar("_Setup", bound=ScheduleSetup)

class SymmetryBreaking(Generic[F]):
    """
    Base class for symmetry breaking constraints.

    Symmetry breaking constraints are used to reduce the search space of the 
    solver by eliminating symmetric solutions.
    For example, in parallel machine scheduling, if two machines are identical,
    we can add a constraint that forces the first task to be assigned to the
    first machine, which breaks the initial symmetry between the machines.
    """

    def __init_subclass__(
        cls: type["SymmetryBreaking[F]"],
        register: bool = True,
    ) -> None:
        if not register:
            return

        for base in get_original_bases(cls):
            if get_origin(base) is not SymmetryBreaking:
                continue

            args = get_args(base)
            if len(args) != 1:
                raise TypeError(
                    f"SymmetryBreaking subclasses must specify a single type  "
                    f"argument representing the formulation type. Got {args}."
                )
            
            formulation_type = args[0]
            if not issubclass(formulation_type, Formulation):
                raise TypeError(
                    f"SymmetryBreaking type argument must be a subclass of "
                    f"Formulation. Got {formulation_type}."
                )

            formulation_type.register_symmetry_breaking(cls)

    def is_appliable(self, env: SchedulingEnv) -> bool:
        """
        Check if the symmetry breaking constraint is appliable to the given environment.
        """
        raise NotImplementedError
    
    def apply(self, formulation: F, env: SchedulingEnv) -> None:
        """
        Apply the symmetry breaking constraint to the given environment.
        """
        raise NotImplementedError

class Formulation(ABC):
    _name: ClassVar[str]

    _setup_registry: ClassVar[dict[type[ScheduleSetup], Exporter[Any, Any]]]
    _constraint_registry: ClassVar[dict[type[Constraint], Exporter[Any, Any]]]
    _objective_registry: ClassVar[dict[type[Objective], VariableExporter[Any, Any]]]
    _symmetry_breaking_registry: ClassVar[list[SymmetryBreaking[Self]]]

    def __init_subclass__(cls, formulation_name: str) -> None:
        cls._name = formulation_name
        formulations[formulation_name] = cls

        cls._setup_registry = {}
        cls._constraint_registry = {}
        cls._objective_registry = {}
        cls._symmetry_breaking_registry = []

    @classmethod
    def register_setup(
        cls: type[F], setup: type[_Setup]
    ) -> Decorator[Exporter[F, _Setup]]:
        def decorator(fn: Exporter[F, _Setup]) -> Exporter[F, _Setup]:
            cls._setup_registry[setup] = fn
            return fn

        return decorator

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
            fn: VariableExporter[F, _Objective]
        ) -> VariableExporter[F, _Objective]:
            cls._objective_registry[objective] = fn
            return fn

        return decorator

    @classmethod
    def register_symmetry_breaking(
        cls: type[F], symmetry_breaking: type[SymmetryBreaking[F]]
        ) -> None:
        cls._symmetry_breaking_registry.append(symmetry_breaking())

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


    def build(self: Self, env: SchedulingEnv, symmetry_break: bool = False) -> None:
        """
        Build the model for the scheduling problem.

        This method initializes the model based on the current instance,
        state, the setup, constraints and objective.
        """
        state = env.state

        if symmetry_break:
            for symmetry_breaking in self._symmetry_breaking_registry:
                if symmetry_breaking.is_appliable(env):
                    symmetry_breaking.apply(self, env)

        setup = env.setup
        if type(setup) not in self._setup_registry:
            raise ValueError(
                f"Setup type '{type(setup).__name__}' is not registered for "
                f"formulation '{type(self).__name__}'."
            )

        self._setup_registry[type(setup)](self, state, setup)

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
        return f"Formulation(name={self._name})"
