from typing import Any
from collections.abc import Iterable

from cpscheduler.environment.utils import convert_to_list

from cpscheduler.environment.constants import TaskID, Time, Int
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import PassiveConstraint


class PreemptionConstraint(PassiveConstraint):
    """
    Preemption constraint for the scheduling environment.
    This constraint allows tasks to be preempted, meaning they can be interrupted
    and resumed later.

    Arguments:
        name: Optional[str] = None
            An optional name for the constraint.

    Note:
        This constraint is a placeholder and does not implement any specific logic
        for preemption. It serves as a marker to indicate that preemption is allowed
        in the scheduling environment, following the convention used in scheduling literature.

        Another way to provide to the environment that preemption is allowed is to set
        the `allow_preemption` flag in the `SchedulingEnv` initialization.
    """

    task_ids: list[TaskID]
    all_tasks: bool
    preemption_tag: str

    def __init__(self, task_ids: Iterable[Int] | str | None = None) -> None:
        self.preemption_tag = ""
        self.all_tasks = False
        self.task_ids = []

        if task_ids is None:
            self.all_tasks = True

        elif isinstance(task_ids, str):
            self.preemption_tag = task_ids

        else:
            self.task_ids = convert_to_list(task_ids, TaskID)

    def __reduce__(self) -> Any:
        task_ids = (
            self.preemption_tag
            if self.preemption_tag
            else (self.task_ids if not self.all_tasks else None)
        )

        return (
            self.__class__,
            (task_ids,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        if self.all_tasks:
            for task_id in range(state.n_tasks):
                state.instance.preemptive[task_id] = True

        elif self.preemption_tag:
            preemption_values = convert_to_list(
                state.instance.task_instance[self.preemption_tag], bool
            )

            for task_id, is_preemptive in enumerate(preemption_values):
                state.instance.preemptive[task_id] = is_preemptive

        else:
            for task_id in self.task_ids:
                state.instance.preemptive[task_id] = True

    def get_entry(self) -> str:
        return "prmp"


class OptionalityConstraint(PassiveConstraint):
    """
    Makes tasks optional in the scheduling environment.
    Tasks marked as optional are treated equally to regular tasks, but they can be
    left unscheduled without affecting the feasibility of the overall schedule.

    Arguments:
        task_ids: Iterable[int] | None
            A list of task IDs to be marked as optional. If None, all tasks are marked as optional.
    """

    task_ids: list[TaskID]
    all_tasks: bool

    def __init__(self, task_ids: Iterable[Int] | str | None = None) -> None:
        self.optionality_tag = ""
        self.all_tasks = False
        self.task_ids = []

        if task_ids is None:
            self.all_tasks = True

        elif isinstance(task_ids, str):
            self.optionality_tag = task_ids

        else:
            self.task_ids = convert_to_list(task_ids, TaskID)

    def __reduce__(self) -> Any:
        task_ids = (
            self.optionality_tag
            if self.optionality_tag
            else (self.task_ids if not self.all_tasks else None)
        )

        return (
            self.__class__,
            (task_ids,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        if self.all_tasks:
            for task_id in range(state.n_tasks):
                state.instance.optional[task_id] = True

        elif self.optionality_tag:
            optional_values = convert_to_list(
                state.instance.task_instance[self.optionality_tag], bool
            )

            for task_id, is_optional in enumerate(optional_values):
                state.instance.optional[task_id] = is_optional

        else:
            for task_id in self.task_ids:
                state.instance.optional[task_id] = True

    def get_entry(self) -> str:
        return "opt"


class ConstantProcessingTime(PassiveConstraint):
    """
    Constant processing time constraint for the scheduling environment.

    This constraint enforces that all tasks have the same processing time, which is defined
    as a constant value.

    Arguments:
        processing_time: int
            The constant processing time for all tasks.

        name: Optional[str] = None
            An optional name for the constraint.
    """

    processing_time: Time

    def __init__(self, processing_time: Int = 1):
        self.processing_time = Time(processing_time)

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.processing_time,),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        instance = state.instance

        for task_id in range(state.n_tasks):
            for machine in state.instance.processing_times[task_id]:
                instance.set_processing_time(
                    task_id, machine, self.processing_time
                )

    def get_entry(self) -> str:
        return f"p_j={self.processing_time}"
