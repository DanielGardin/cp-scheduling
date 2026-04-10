from typing import Any

from math import expm1

from cpscheduler.environment.utils import convert_to_list
from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.objectives.base import Objective, RegularObjective
from cpscheduler.environment.objectives.makespan import makespan_

class TotalCompletionTime(RegularObjective):
    """
    The total completion time objective function, which aims to minimize the sum
    of completion times of all tasks.
    """
    _job_completion: dict[TaskID, Time]

    def __init__(self, minimize: bool = True) -> None:
        super().__init__(minimize)
        self._job_completion = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.minimize,),
            (self._job_completion,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._job_completion,) = state

    def reset(self, state: ScheduleState) -> None:
        self._job_completion.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        self._job_completion[job_id] = state.time

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_completion.values()))
    
    def __call__(self, state: ScheduleState) -> float:
        return sum(
            makespan_(state, tasks)
            for tasks in state.instance.job_tasks
        )

    def get_entry(self) -> str:
        return "ΣC_j"


class WeightedCompletionTime(Objective):
    """
    The weighted completion time objective function, which aims to minimize the
    weighted sum of completion times of all tasks.
    Each task has a weight associated with it, and the objective function is the
    sum of the completion times multiplied by their respective weights.
    """

    _weighted_job_completion: dict[TaskID, float]

    weights_tag: str
    job_weights: list[float]

    def __init__(
        self,
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.weights_tag = job_weights
        self._weighted_job_completion = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.weights_tag, self.minimize,),
            (self._weighted_job_completion, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._weighted_job_completion, self.job_weights) = state

    @property
    def regular(self) -> bool:
        return all(weight >= 0 for weight in self.job_weights)

    def initialize(self, state: ScheduleState) -> None:
        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weights_tag], float
        )

    def reset(self, state: ScheduleState) -> None:
        self._weighted_job_completion.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        weight = self.job_weights[job_id]

        self._weighted_job_completion[job_id] = weight * float(state.time)

    def get_current(self, state: ScheduleState) -> float:
        return sum(self._weighted_job_completion.values())

    def __call__(self, state: ScheduleState) -> float:
        job_weights = self.job_weights

        return sum(
            weight * makespan_(state, tasks)
            for weight, tasks in zip(job_weights, state.instance.job_tasks)
        )

    def get_entry(self) -> str:
        return "Σw_jC_j"

class DiscountedTotalCompletionTime(RegularObjective):
    _discounted_job_completion: dict[TaskID, float]
    discount_factor: float

    def __init__(
        self,
        discount_factor: float = 0.99,
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.discount_factor = discount_factor
        self._discounted_job_completion = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.discount_factor, self.minimize,),
            (self._discounted_job_completion,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._discounted_job_completion,) = state

    def reset(self, state: ScheduleState) -> None:
        self._discounted_job_completion.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]

        self._discounted_job_completion[job_id] = -expm1(state.time)

    def get_current(self, state: ScheduleState) -> float:
        return sum(self._discounted_job_completion.values())

    def __call__(self, state: ScheduleState) -> float:
        return - sum(
            expm1(makespan_(state, tasks))
            for tasks in state.instance.job_tasks
        )

    def get_entry(self) -> str:
        return f"Σ(1 - e^(-{self.discount_factor}C_j))"

class TotalFlowTime(RegularObjective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all
    tasks. Flow time is defined as the difference between the completion time and the release time.
    """

    _job_flow: dict[TaskID, Time]

    release_tag: str
    release_times: list[Time]

    def __init__(
        self,
        release_times: str = "release_time",
        minimize: bool = True
    ) -> None:
        super().__init__(minimize)
        self.release_tag = release_times
        self._job_flow = {}

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.release_tag, self.minimize,),
            (self._job_flow,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._job_flow,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.release_dates = convert_to_list(
            state.instance.task_instance[self.release_tag], Time
        )

    def reset(self, state: ScheduleState) -> None:
        self._job_flow.clear()

    def on_task_completed(
        self, task_id: TaskID, machine_id: MachineID, state: ScheduleState
    ) -> None:
        job_id = state.instance.job_ids[task_id]
        release_time = self.release_times[job_id]

        self._job_flow[job_id] = state.time - release_time

    def get_current(self, state: ScheduleState) -> float:
        return float(sum(self._job_flow.values()))

    def __call__(self, state: ScheduleState) -> float:
        return sum(
            makespan_(state, tasks) - float(release_time)
            for release_time, tasks in zip(self.release_dates, state.instance.job_tasks)
        )

    def get_entry(self) -> str:
        return "ΣF_j"