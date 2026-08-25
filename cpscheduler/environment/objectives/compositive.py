"""Composed objective functions."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import Float, MachineID, TaskID, Time
from cpscheduler.environment.instance import ProblemInstance
from cpscheduler.environment.objectives.base import Objective
from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.utils.general import convert_to_list


class ComposedObjective(Objective):
    """A composed objective function that combines multiple objectives with coefficients.

    The overall objective value is a weighted sum of the individual objectives.
    A composed objective is regular if all non-zero-coefficient components are regular
    and no regular component has a negative coefficient.
    """

    objectives: list[Objective]
    coefficients: list[float]

    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Iterable[Float] | None = None,
        minimize: bool = True,
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

        minimize: bool, optional
            Whether the composed objective should be minimized (True) or maximized (False).
            Default is True (i.e., minimize).

        """
        super().__init__(minimize)

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

    @property
    @override
    def regular(self) -> bool:
        return all(
            (coef == 0 or objective.regular)
            and not (coef < 0 and objective.regular)
            for objective, coef in zip(
                self.objectives, self.coefficients, strict=False
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
            objective.on_assignment(task_id, machine_id, state)

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
    def compute(self, state: ScheduleState) -> float:
        return sum(
            coefficient * objective.compute(state)
            for objective, coefficient in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

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
