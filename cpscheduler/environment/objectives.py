"""
objectives.py

This module defines the objective functions used in the scheduling environment.
Objective functions are used to evaluate the performance of a scheduling algorithm
and can be used to guide the search for an optimal schedule by providing a numerical value
that represents the quality of the schedule.
"""

from typing import Any, ClassVar
from collections.abc import Iterable

from mypy_extensions import mypyc_attr

from math import expm1

from cpscheduler.utils.list_utils import convert_to_list
from cpscheduler.environment.constants import TaskID, Float
from cpscheduler.environment.state import ScheduleState

objectives: dict[str, type["Objective"]] = {}


@mypyc_attr(allow_interpreted_subclasses=True)
class Objective:
    """
    Base class for all objective functions in the scheduling environment.

    Objective functions are used to evaluate the performance of a scheduling
    algorithm.
    They can be used to guide the search for an optimal schedule by providing a
    numerical value that represents the quality of the schedule.
    """

    _regular: ClassVar[bool] = False

    minimize: bool

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        objectives[cls.__name__] = cls

    def __init__(self, minimize: bool = True) -> None:
        self.minimize = minimize

    @property
    def regular(self) -> bool:
        "The objective is regular, when it is non-decreasing w.r.t completion times."
        return self._regular

    def __repr__(self) -> str:
        sense = "minimize" if self.minimize else "maximize"

        return f"{self.__class__.__name__}(sense={sense})"

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.minimize,),
            (),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        pass

    def initialize(self, state: ScheduleState) -> None:
        "Initialize the objective with the given schedule state."

    def get_current(self, state: ScheduleState) -> float:
        """
        Get the current value of the objective function. This is useful for checking
        the performance of the scheduling algorithm along the episode.
        """
        return 0.0

    def __call__(self, state: ScheduleState) -> float:
        "Call the objective function to get the current value."
        return self.get_current(state)

    def get_entry(self) -> str:
        "Produce the γ entry for the constraint."
        return ""


class ComposedObjective(Objective):
    """
    A composed objective function that combines multiple objectives with coefficients.

    This objective function allows for the combination of multiple objectives into a
    single objective function. Each objective can have a coefficient that scales its
    contribution to the overall objective value. The overall objective value is computed
    as a weighted sum of the individual objectives.

    Arguments:
        objectives: Iterable[Objective]
            An iterable of `Objective` instances to be combined.

        coefficients: Iterable[float], optional
            An iterable of coefficients for each objective. If not provided, all objectives
            are assumed to have a coefficient of 1.0.

        minimize: bool, default=True
            Whether to minimize or maximize the objective function.
    """

    objectives: list[Objective]
    coefficients: list[float]

    def __init__(
        self,
        objectives: Iterable[Objective],
        coefficients: Iterable[Float] | None = None,
        minimize: bool = True,
    ):
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
    def regular(self) -> bool:
        "A composed objective is regular if all its component objectives are regular."
        for objective, coefficient in zip(self.objectives, self.coefficients):
            if coefficient != 0 and not objective.regular:
                return False

            if coefficient < 0 and objective.regular:
                return False

        return True

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.objectives, self.coefficients, self.minimize),
            (),
        )

    def initialize(self, state: ScheduleState) -> None:
        for objective in self.objectives:
            objective.initialize(state)

    def get_current(self, state: ScheduleState) -> float:
        current_value = 0.0

        for objective, coefficient in zip(self.objectives, self.coefficients):
            current_value += coefficient * objective.get_current(state)

        return current_value

    def get_entry(self) -> str:
        terms: list[str] = []

        for coef, objective in zip(self.coefficients, self.objectives):
            if coef == 0:
                continue

            abs_coef = abs(coef)
            if abs_coef == 1:
                term = objective.get_entry()

            else:
                coef_str = (
                    str(int(abs_coef))
                    if abs_coef.is_integer()
                    else f"{abs_coef:.2f}"
                )
                term = f"{coef_str} {objective.get_entry()}"

            if not terms:
                terms.append(f"- {term}" if coef < 0 else term)

            else:
                terms.append(f"{'-' if coef < 0 else '+'} {term}")

        return " ".join(terms) if terms else "0"


def _makespan(state: ScheduleState, tasks: Iterable[TaskID]) -> float:
    "Compute the makespan of a set of tasks."
    max_end_time = 0.0

    for task in tasks:
        if not state.is_completed(task):
            continue

        end_time = float(state.get_end_ub(task))

        if end_time > max_end_time:
            max_end_time = end_time

    return max_end_time


class Makespan(Objective):
    """
    Classic makespan objective function, which aims to minimize the time at
    which all tasks are completed.
    """

    _regular = True

    def get_current(self, state: ScheduleState) -> float:
        return _makespan(state, state.runtime_state.completed_tasks)

    def get_entry(self) -> str:
        return "C_max"


class TotalCompletionTime(Objective):
    """
    The total completion time objective function, which aims to minimize the sum
    of completion times of all tasks.
    """

    _regular = True

    def get_current(self, state: ScheduleState) -> float:
        total_completion_time = 0.0

        for tasks in state.instance.job_tasks:
            job_completion = _makespan(state, tasks)
            total_completion_time += job_completion

        return total_completion_time

    def get_entry(self) -> str:
        return "ΣC_j"


class WeightedCompletionTime(Objective):
    """
    The weighted completion time objective function, which aims to minimize the
    weighted sum of completion times of all tasks.
    Each task has a weight associated with it, and the objective function is the
    sum of the completion times multiplied by their respective weights.
    """

    _regular = True

    weights_tag: str
    job_weights: list[float]

    def __init__(
        self,
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.weights_tag = job_weights
        self.job_weights = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.weights_tag, self.minimize),
            (self.job_weights,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.job_weights,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weights_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        weighted_completion_time = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            weight = self.job_weights[job_id]
            job_completion = _makespan(state, tasks)

            weighted_completion_time += weight * float(job_completion)

        return weighted_completion_time

    def get_entry(self) -> str:
        return "Σw_jC_j"


class DiscountedCompletionTime(Objective):

    _regular = True

    discount_factor: float

    weights_tag: str | None
    job_weights: list[float]

    def __init__(
        self,
        discount_factor: float = 0.99,
        job_weights: str | None = None,
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.discount_factor = discount_factor

        self.weights_tag = job_weights
        self.job_weights = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.discount_factor, self.weights_tag, self.minimize),
            (self.job_weights,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.job_weights,) = state

    def initialize(self, state: ScheduleState) -> None:
        if self.weights_tag is not None:
            self.job_weights = convert_to_list(
                state.instance.task_instance[self.weights_tag], float
            )

    def get_current(self, state: ScheduleState) -> float:
        discounted_completion_time = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            weight = (
                self.job_weights[job_id]
                if self.weights_tag is not None
                else 1.0
            )
            job_completion = _makespan(state, tasks)

            discounted_completion_time -= weight * expm1(
                -self.discount_factor * float(job_completion)
            )

        return discounted_completion_time

    def get_entry(self) -> str:
        if self.weights_tag is not None:
            return f"Σw_j(1 - e^(-{self.discount_factor}C_j))"

        return f"Σ(1 - e^(-{self.discount_factor}C_j))"


class MaximumLateness(Objective):
    """
    The maximum lateness objective function, which aims to minimize the maximum
    lateness of all tasks.
    Lateness is defined as the difference between the completion time and the
    due date.
    """

    _regular = True

    due_tag: str
    due_dates: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        max_lateness = 0.0

        for job_id, tasks in enumerate(state.instance.job_tasks):
            job_completion = _makespan(state, tasks)
            job_lateness = job_completion - self.due_dates[job_id]

            if max_lateness < job_lateness:
                max_lateness = job_lateness

        return max_lateness

    def get_entry(self) -> str:
        return "L_max"


class TotalTardiness(Objective):
    """
    The total tardiness objective function, which aims to minimize the sum of
    tardiness of all tasks.
    Tardiness is defined as the difference between the completion time and the
    due date, if the task is completed late.
    """

    _regular = True

    due_tag: str
    due_dates: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        total_tardiness = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            job_tardiness = (
                job_completion - due_date if job_completion > due_date else 0
            )

            total_tardiness += job_tardiness

        return total_tardiness

    def get_entry(self) -> str:
        return "ΣT_j"


class WeightedTardiness(Objective):
    """
    The weighted tardiness objective function, which aims to minimize the weighted sum of tardiness
    of all tasks. Tardiness is defined as the difference between the completion time and the due
    date, if the task is completed late.
    """

    _regular = True

    due_tag: str
    weight_tag: str
    due_dates: list[float]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)

        self.due_tag = due_dates
        self.weight_tag = job_weights

        self.due_dates = []
        self.job_weights = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.weight_tag, self.minimize),
            (self.due_dates, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.due_dates, self.job_weights = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )
        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        weighted_tardiness = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            weight = self.job_weights[job_id]
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            job_tardiness = (
                float(job_completion - due_date)
                if job_completion > due_date
                else 0.0
            )

            weighted_tardiness += weight * job_tardiness

        return weighted_tardiness

    def get_entry(self) -> str:
        return "Σw_jT_j"


class TotalEarliness(Objective):
    """
    The total earliness objective function, which aims to minimize the sum of earliness of all
    tasks. Earliness is defined as the difference between the due date and the completion time,
    if the task is completed early.
    """

    due_tag: str
    due_dates: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        total_earliness = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            job_earliness = (
                due_date - job_completion if job_completion < due_date else 0.0
            )

            total_earliness += job_earliness

        return total_earliness

    def get_entry(self) -> str:
        return "ΣE_j"


class WeightedEarliness(Objective):
    """
    The weighted earliness objective function, which aims to minimize the weighted sum of earliness of all tasks.
    Earliness is defined as the difference between the due date and the completion time, if the task is completed early.
    """

    due_tag: str
    weight_tag: str
    due_dates: list[float]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.weight_tag = job_weights

        self.due_dates = []
        self.job_weights = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.weight_tag, self.minimize),
            (self.due_dates, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.due_dates, self.job_weights = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )
        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        weighted_earliness = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            weight = self.job_weights[job_id]
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            job_earliness = (
                float(due_date - job_completion)
                if job_completion < due_date
                else 0
            )

            weighted_earliness += weight * job_earliness

        return weighted_earliness

    def get_entry(self) -> str:
        return "Σw_jE_j"


class TotalTardyJobs(Objective):
    """
    The total tardy jobs objective function, which aims to minimize the number of tardy jobs.
    A job is considered tardy if its completion time is greater than its due date.
    """

    _regular = True

    due_tag: str
    due_dates: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.due_dates = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.minimize),
            (self.due_dates,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.due_dates,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        tardy_jobs = 0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            tardy = 1 if job_completion > due_date else 0
            tardy_jobs += tardy

        return tardy_jobs

    def get_entry(self) -> str:
        return "ΣU_j"


class WeightedTardyJobs(Objective):
    """
    The weighted tardy jobs objective function, which aims to minimize the weighted sum of tardy
    jobs. A job is considered tardy if its completion time is greater than its due date.
    """

    _regular = True

    due_tag: str
    weight_tag: str
    due_dates: list[float]
    job_weights: list[float]

    def __init__(
        self,
        due_dates: str = "due_date",
        job_weights: str = "weight",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.due_tag = due_dates
        self.weight_tag = job_weights

        self.due_dates = []
        self.job_weights = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.due_tag, self.weight_tag, self.minimize),
            (self.due_dates, self.job_weights),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.due_dates, self.job_weights = state

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], float
        )
        self.job_weights = convert_to_list(
            state.instance.task_instance[self.weight_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        weighted_tardy_jobs = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            weight = self.job_weights[job_id]
            due_date = self.due_dates[job_id]
            job_completion = _makespan(state, tasks)

            tardy = weight if job_completion > due_date else 0.0
            weighted_tardy_jobs += tardy

        return weighted_tardy_jobs

    def get_entry(self) -> str:
        return "Σw_jU_j"


class TotalFlowTime(Objective):
    """
    The total flow time objective function, which aims to minimize the sum of flow times of all
    tasks. Flow time is defined as the difference between the completion time and the release time.
    """

    _regular = True

    release_tag: str
    release_times: list[float]

    def __init__(
        self,
        release_times: str = "release_time",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.release_tag = release_times
        self.release_times = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.release_tag, self.minimize),
            (self.release_times,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.release_times,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.release_times = convert_to_list(
            state.instance.task_instance[self.release_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        total_flowtime = 0.0
        for job_id, tasks in enumerate(state.instance.job_tasks):
            release_time = self.release_times[job_id]
            job_completion = _makespan(state, tasks)

            job_flowtime = (
                job_completion - release_time
                if job_completion > release_time
                else 0.0
            )

            total_flowtime += job_flowtime

        return total_flowtime

    def get_entry(self) -> str:
        return "ΣF_j"


class RejectionCost(Objective):
    """
    The rejection cost objective function, which aims to minimize the total cost
    of rejected jobs.
    A job is considered rejected when it is optional and either:
    - Is unfeasible to schedule.
    - Was not scheduled when the environment reaches a terminal state.
    """

    _regular = False

    cost_tag: str
    rejection_costs: list[float]

    def __init__(
        self,
        cost_tag: str = "rejection_cost",
        minimize: bool = True,
    ):
        super().__init__(minimize)
        self.cost_tag = cost_tag
        self.rejection_costs = []

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.cost_tag, self.minimize),
            (self.rejection_costs,),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self.rejection_costs,) = state

    def initialize(self, state: ScheduleState) -> None:
        self.rejection_costs = convert_to_list(
            state.instance.task_instance[self.cost_tag], float
        )

    def get_current(self, state: ScheduleState) -> float:
        total_rejection_cost = 0.0

        for task_id, opt in enumerate(state.instance.optional):
            if opt and not state.is_fixed(task_id):
                total_rejection_cost += self.rejection_costs[task_id]

        return total_rejection_cost
