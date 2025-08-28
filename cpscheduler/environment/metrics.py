from typing import Any, Protocol, TypeVar
from collections.abc import Mapping, Iterable

from cpscheduler.utils.typing_utils import is_iterable_type

from .tasks import Tasks
from .data import SchedulingData
from ._common import Int, TASK_ID, TIME


def machine_utilization(
    time: int, tasks: Tasks, data: SchedulingData, objective: float
) -> float:
    """
    Calculate the percentage of time that machines are utilized during the scheduling period.
    """

    total_time = time
    busy_time = 0
    for task in tasks.fixed_tasks:
        busy_time += task.get_processed_time(time)

    return data.n_machines * busy_time / total_time if total_time > 0 else 1


def max_preemptions(
    time: int, tasks: Tasks, data: SchedulingData, objective: float
) -> int:
    "Calculate the maximum number of preemption switches that occurred during the scheduling period."
    max_switches = 0
    for task in tasks:
        n_switches = task.n_parts - 1

        if n_switches > max_switches:
            max_switches = n_switches

    return max_switches


class ReferenceScheduleMetrics:
    """
    This is a metric class that calculates metrics based on a reference schedule.
    It can be initialized with a reference schedule, which can be a mapping of task IDs to
    their scheduled start times, an iterable of tuples (task_id, start_time), or a
    string which is used as a tag to fetch the reference schedule from the data.

    The available metrics include:
    - total_displacement_distance: The total distance between the scheduled and actual start times of tasks.

    - order_preservation: The ratio of tasks that maintain their order in the reference schedule.

    - hamming_accuracy: The ratio of tasks that are in the same position as in the reference schedule.

    - kendall_tau: The Kendall Tau distance between the reference schedule and the actual schedule.
    """

    reference_schedule: dict[TASK_ID, TIME]

    def __init__(
        self,
        reference_schedule: Mapping[Int, Int] | Iterable[tuple[Int, Int]] | str,
        permutation_based: bool = False,
    ):
        self.tag = ""
        self.permutation_based = permutation_based
        self.force_import = True

        if isinstance(reference_schedule, str):
            self.tag = reference_schedule
            self.reference_schedule = {}

        elif is_iterable_type(reference_schedule, tuple):
            self.reference_schedule = {
                TASK_ID(task_id): TIME(time) for task_id, time in reference_schedule
            }

        else:
            self.reference_schedule = {
                TASK_ID(task_id): TIME(time)
                for task_id, time in reference_schedule.items()
            }

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            self.__class__,
            (self.tag if self.tag else self.reference_schedule, self.permutation_based),
        )

    def __call__(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> dict[str, float]:
        self.force_import = False

        metrics = {
            "mean_displacement_distance": self.mean_displacement_distance(
                time, tasks, data, objective
            ),
            "order_preservation": self.order_preservation(time, tasks, data, objective),
        }

        if self.permutation_based:
            metrics.update(
                {
                    "hamming_accuracy": self.hamming_accuracy(
                        time, tasks, data, objective
                    ),
                    "kendall_tau": self.kendall_tau(time, tasks, data, objective),
                }
            )

        self.force_import = True

        return metrics

    def import_data(self, data: SchedulingData) -> None:
        if self.tag:
            schedule_info = data.get_task_level_data(self.tag)

            self.reference_schedule = {
                TASK_ID(task_id): TIME(start_time)
                for task_id, start_time in enumerate(schedule_info)
            }

    def mean_displacement_distance(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the mean displacement distance of the reference schedule.
        The displacement distance is the sum of the absolute differences between
        the scheduled and actual start times of each task.
        """
        if self.force_import:
            self.import_data(data)

        distance = 0
        count = 0

        for task_id, reference_time in self.reference_schedule.items():
            task = tasks[task_id]

            if task.is_fixed():
                actual_time = task.get_start()

                # abs does not seem to supported by mypyc yet
                distance += (
                    reference_time - actual_time
                    if reference_time > actual_time
                    else actual_time - reference_time
                )
                count += 1

        return distance / count if count > 0 else 0.0

    def order_preservation(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the order preservation metric based on the reference schedule.
        This metric is the ratio of the number of tasks that maintain their order
        in the reference schedule to the total number of tasks.
        """
        if self.force_import:
            self.import_data(data)

        reference_count = 0
        preserved_count = 0

        for task_i in self.reference_schedule:
            if not tasks[task_i].is_fixed():
                continue

            ref_start_i = self.reference_schedule[task_i]
            actual_start_i = tasks[task_i].get_start()

            for task_j in self.reference_schedule:
                if not tasks[task_j].is_fixed():
                    continue

                ref_start_j = self.reference_schedule[task_j]
                if ref_start_j >= ref_start_i:
                    continue

                reference_count += 1

                actual_start_j = tasks[task_j].get_start()
                if actual_start_j < actual_start_i:
                    preserved_count += 1

        return preserved_count / reference_count if reference_count > 0 else 1.0

    # The following are useful when the schedule is a permutation of the tasks
    def _get_permutations(self, tasks: Tasks) -> tuple[list[TASK_ID], list[TASK_ID]]:
        reference_perm = sorted(
            [
                task_id
                for task_id in self.reference_schedule
                if tasks[task_id].is_fixed()
            ],
            key=lambda task_id: self.reference_schedule[task_id],
        )

        actual_perm = sorted(
            reference_perm, key=lambda task_id: tasks[task_id].get_start()
        )

        return reference_perm, actual_perm

    def hamming_accuracy(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the Hamming similarity of the schedule permutation compared to the reference
        schedule. This metric is the number of tasks that are in the same position
        as in the reference schedule.
        """
        if self.force_import:
            self.import_data(data)

        reference_perm, actual_perm = self._get_permutations(tasks)

        hamming_count = 0
        for ref_task, act_task in zip(reference_perm, actual_perm):
            if ref_task == act_task:
                hamming_count += 1

        return hamming_count / len(reference_perm) if reference_perm else 1.0

    def kendall_tau(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the Kendall Tau distance between the reference schedule and the actual schedule.
        This metric is the number of discordant pairs in the schedule.
        """
        if self.force_import:
            self.import_data(data)

        reference_perm, actual_perm = self._get_permutations(tasks)
        n = len(reference_perm)

        concordant = 0
        discordant = 0
        ties_ref = 0
        ties_act = 0

        for i in range(n):
            for j in range(i + 1, n):
                ref_i = reference_perm[i]
                ref_j = reference_perm[j]
                act_i = actual_perm[i]
                act_j = actual_perm[j]

                if ref_i == ref_j and act_i == act_j:
                    continue

                elif ref_i == ref_j:
                    ties_ref += 1

                elif act_i == act_j:
                    ties_act += 1

                else:
                    concordant += (ref_i - ref_j) * (act_i - act_j) > 0
                    discordant += (ref_i - ref_j) * (act_i - act_j) < 0

        denominator = (
            (concordant + discordant + ties_ref) * (concordant + discordant + ties_act)
        ) ** 0.5

        return (concordant - discordant) / denominator if denominator > 0 else 1.0


class OptimalReferenceMetrics(ReferenceScheduleMetrics):
    """
    This class is a specialized version of ReferenceScheduleMetrics that uses the optimal schedule
    as the reference schedule. It is used to calculate any reference metrics and additionally
    provides metrics that are based on the optimal schedule.

    The additional metrics are:
    """

    def __init__(
        self,
        optimal_schedule: (
            Mapping[Int, Int] | Iterable[tuple[Int, Int]] | str | None
        ) = None,
        optimal_value: float = 0.0,
        permutation_based: bool = False,
    ):
        optimal_schedule = optimal_schedule if optimal_schedule else {}

        super().__init__(optimal_schedule, permutation_based)

        self.optimal_value = optimal_value

    def __reduce__(self) -> tuple[Any, ...]:
        _, (schedule, perm) = super().__reduce__()

        return (self.__class__, (schedule, self.optimal_value, perm))

    def regret(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the regret of the current schedule compared to the optimal schedule.
        Regret is defined as the difference between the optimal value and the current objective value.
        """
        return self.optimal_value - objective

    def optimality_gap(
        self, time: int, tasks: Tasks, data: SchedulingData, objective: float
    ) -> float:
        """
        Calculate the optimality gap of the current schedule compared to the optimal schedule.
        The optimality gap is defined as the ratio of the regret to the optimal value.
        """
        return objective / (self.optimal_value + 1e-9) - 1.0
