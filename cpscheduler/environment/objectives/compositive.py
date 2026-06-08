"""Composed objective functions."""

from collections.abc import Iterable
from typing import override

from cpscheduler.environment.constants import Float, MachineID, TaskID
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
            (coefficient == 0 or objective.regular)
            and not (coefficient < 0 and objective.regular)
            for objective, coefficient in zip(
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

    @override
    def on_task_started(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_task_started(task_id, machine_id, state)

    @override
    def on_task_paused(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_task_paused(task_id, machine_id, state)

    @override
    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_task_completed(task_id, machine_id, state)

    @override
    def on_task_machine_infeasible(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        for objective in self.objectives:
            objective.on_task_machine_infeasible(task_id, machine_id, state)

    @override
    def get_current(self, state: ScheduleState) -> float:
        return sum(
            coefficient * objective.get_current(state)
            for objective, coefficient in zip(
                self.objectives, self.coefficients, strict=False
            )
        )

    @override
    def __call__(self, state: ScheduleState) -> float:
        return sum(
            coefficient * objective(state)
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
