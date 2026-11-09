"""Composed objective functions."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Float, MachineID, TaskID, Time
from cpscheduler.environment.instance import Feature, ProblemInstance
from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class ComposedObjective(Objective):
    """A composed objective function that combines multiple objectives with coefficients.

    The composition is the function sum_i c_i * f_i: the coefficients carry the
    direction of each term, and the terms' `default_sense` is not used. Its own
    default sense is minimization.

    It is regular when every term with a positive coefficient is regular and
    every term with a negative coefficient is anti-regular, and anti-regular in
    the opposite case. It is backtrack-safe when all its terms are.
    """

    objectives: list[Objective]
    coefficients: list[float]

    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Iterable[Float] | None = None,
    ):
        """Initialize the ComposedObjective.

        Parameters
        ----------
        objectives: Iterable[Objective]
            The list of objective components to be combined.

        coefficients: Iterable[Float] | None, optional
            The coefficients for each objective component.
            If None is provided, all coefficients will be set to 1.0.
            Default to None.

        """
        super().__init__()

        self.objectives = list(objectives)
        self.coefficients = (
            [1.0] * len(self.objectives)
            if coefficients is None
            else convert_to_list(coefficients, float)
        )

        if len(self.coefficients) != len(self.objectives):
            raise ValueError(
                "The number of coefficients must match the number of objectives."
            )

    @override
    def get_features(self) -> list[Feature]:
        features = super().get_features()

        for objective in self.objectives:
            features.extend(objective.get_features())

        return features

    @property
    @override
    def backtrack_safe(self) -> bool:
        return all(objective.backtrack_safe for objective in self.objectives)

    @property
    @override
    def regular(self) -> bool:
        # A negative coefficient reverses the direction of its term
        return all(
            coef == 0
            or (objective.regular if coef > 0 else objective.anti_regular)
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=True
            )
        )

    @property
    @override
    def anti_regular(self) -> bool:
        return all(
            coef == 0
            or (objective.anti_regular if coef > 0 else objective.regular)
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=True
            )
        )

    @override
    def reset(self, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.reset(state)

    @override
    def initialize(self, instance: ProblemInstance) -> None:
        for objective in self.objectives:
            objective.initialize(instance)

    @property
    @override
    def lb(self) -> float:
        return sum(
            coef * (objective.lb if coef >= 0 else objective.ub)
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

    @property
    @override
    def ub(self) -> float:
        return sum(
            coef * (objective.ub if coef >= 0 else objective.lb)
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

    @property
    @override
    def value(self) -> float:
        return sum(
            coef * objective.value
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

    @override
    def on_assignment(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_assignment(task_id, machine_id, state)

    @override
    def on_start_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_start_lb(task_id, machine_id, state)

    @override
    def on_start_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_start_ub(task_id, machine_id, state)

    @override
    def on_end_lb(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_end_lb(task_id, machine_id, state)

    @override
    def on_end_ub(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_end_ub(task_id, machine_id, state)

    @override
    def on_presence(self, task_id: TaskID, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_presence(task_id, state)

    @override
    def on_absence(self, task_id: TaskID, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_absence(task_id, state)

    @override
    def on_infeasibility(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_infeasibility(task_id, machine_id, state)

    @override
    def on_time_update(self, time: Time, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_time_update(time, state)

    @override
    def on_task_lock(self, task_id: TaskID, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_task_lock(task_id, state)

    @override
    def on_task_unlock(self, task_id: TaskID, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_task_unlock(task_id, state)

    @override
    def compute(self, state: ScheduleState) -> float:
        return sum(
            coefficient * objective.compute(state)
            for objective, coefficient in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

    @override
    def on_failure(self, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.on_failure(state)

    @override
    def checkpoint(self, mark: int) -> None:
        for objective in self.objectives:
            objective.checkpoint(mark)

    @override
    def backtrack(
        self, mark: int, changed_tasks: set[TaskID], state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.backtrack(mark, changed_tasks, state)

    @override
    def get_entry(self) -> str:
        terms: list[str] = []

        for coef, objective in zip(
            self.coefficients, self.objectives, strict=False
        ):
            if coef == 0:
                continue

            abs_coef = abs(coef)
            coef_str = (
                str(int(abs_coef))
                if abs_coef.is_integer()
                else f"{abs_coef:.2f}"
            )
            term = (
                objective.get_entry()
                if abs_coef == 1
                else f"{coef_str} {objective.get_entry()}"
            )
            sign = "-" if coef < 0 else "+"
            terms.append(
                f"{sign} {term}"
                if terms
                else (f"- {term}" if coef < 0 else term)
            )

        return " ".join(terms) if terms else "0"

    @override
    @classmethod
    def get_general_entry(cls) -> str:
        return "Σw_kf_k"
