from typing import Any
from copy import deepcopy

from math import sqrt

from cpscheduler.environment.constants import TaskID, Time, MachineID
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.des import ActionType
from cpscheduler.environment.state import ScheduleState


def machine_utilization(state: ScheduleState) -> float:
    """
    Calculate the percentage of time that machines are utilized during the scheduling period.

    When time is 0, the utilization is defined as 100% utilization.
    """
    time = state.time

    total_time = float(time * state.n_machines)
    busy_time: Time = 0
    for history in state.runtime.history:
        for entry in history:
            end_time = entry.end_time
            start_time = entry.start_time

            busy_time += (
                end_time - start_time if end_time <= time else time - start_time
            )

    return float(busy_time) / total_time if total_time > 0 else 1


def num_preemptions(state: ScheduleState) -> int:
    "Calculate the total number of preemption switches that occurred during the scheduling period."
    total_switches = 0
    for history in state.runtime.history:
        total_switches += len(history) - 1

    return total_switches


def max_preemptions(state: ScheduleState) -> int:
    "Calculate the maximum number of preemption switches that occurred during the scheduling period."
    max_switches = 0
    for history in state.runtime.history:
        n_switches = len(history) - 1

        if n_switches > max_switches:
            max_switches = n_switches

    return max_switches


def _count_inversions(arr: list[Any]) -> int:
    if len(arr) < 2:
        return 0

    mid = len(arr) // 2
    left, right = arr[:mid], arr[mid:]
    inversions = _count_inversions(left) + _count_inversions(right)

    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1

        else:
            inversions += len(left) - i
            arr[k] = right[j]
            j += 1

        k += 1

    arr[k:] = left[i:] if i < len(left) else right[j:]

    return inversions


class ReferenceScheduleMetrics:
    """
    Calculates disparity metrics between the reference schedule and the actual schedule.
    These metrics evaluate how closely the constructed schedule follows the reference.

    For now, preemption is not considered in the reference schedule, and only the last start
    and assignment of each task are considered for the metrics.

    The available metrics include:
    > Time reference metrics
    - mean_displacement_distance: The mean distance between the scheduled and actual start times of tasks.
    - early_displacement_ratio: The ratio of tasks that start earlier than in the reference schedule.
    - late_displacement_ratio: The ratio of tasks that start later than in the reference schedule

    > Order reference metrics
    - order_preservation: The ratio of tasks that maintain their order in the reference schedule.
    - hamming_accuracy: The ratio of tasks that are in the same position as in the reference schedule.
    - kendall_tau: The Kendall Tau distance between the reference schedule and the actual schedule.

    > Machine assignment metrics
    - machine_accuracy: The ratio of tasks that are assigned to the same machine as in the reference schedule.

    """

    start_times: dict[TaskID, Time]
    assignments: dict[TaskID, MachineID]

    sorted_start_times: list[tuple[TaskID, Time]]

    # TODO: Ensure that cpy_env time has all scheduled tasks completed.
    def __init__(self, reference_schedule: ActionType, env: SchedulingEnv):
        self.start_times = {}
        self.assignments = {}

        cpy_env = deepcopy(env)
        cpy_env.step(reference_schedule)

        runtime = cpy_env.state.runtime

        for task_id in runtime.completed_tasks:
            self.assignments[task_id] = runtime.get_assignment(task_id)
            self.start_times[task_id] = runtime.get_start(task_id)

        self.sorted_start_times = sorted(
            [(task_id, start) for task_id, start in self.start_times.items()],
            key=lambda x: x[1],
        )

    # Collect all metrics automatically in a single call to avoid redundant calculations
    def __call__(self, state: ScheduleState) -> dict[str, float]:
        metrics = {
            "mean_displacement_distance": self.mean_displacement_distance(
                state
            ),
            "order_preservation": self.order_preservation(state),
            "hamming_accuracy": self.hamming_accuracy(state),
            "kendall_tau": self.kendall_tau(state),
            "machine_accuracy": self.machine_accuracy(state),
        }

        return metrics

    # Time reference metrics

    def mean_displacement_distance(self, state: ScheduleState) -> float:
        """
        Calculate the mean displacement distance of the reference schedule.
        The displacement distance is the sum of the absolute differences between
        the scheduled and actual start times of each task.
        """
        distance = 0
        count = 0

        runtime = state.runtime
        for task_id in runtime.completed_tasks:
            if task_id not in self.start_times:
                continue

            ref_start = self.start_times[task_id]
            actual_start = runtime.get_start(task_id)

            distance += (
                ref_start - actual_start
                if ref_start > actual_start
                else actual_start - ref_start
            )
            count += 1

        return distance / count if count > 0 else 0.0

    def early_displacement_ratio(self, state: ScheduleState) -> float:
        """
        Calculate the early displacement ratio of the reference schedule.
        This metric is the ratio of tasks that start earlier than in the reference schedule
        to the total number of tasks.
        """
        early_count = 0
        count = 0

        runtime = state.runtime
        for task_id in state.runtime.completed_tasks:
            if task_id not in self.start_times:
                continue

            ref_start = self.start_times[task_id]
            actual_start = runtime.get_start(task_id)

            if actual_start < ref_start:
                early_count += 1

            count += 1

        return early_count / count if count > 0 else 1.0

    def late_displacement_ratio(self, state: ScheduleState) -> float:
        """
        Calculate the late displacement ratio of the reference schedule.
        This metric is the ratio of tasks that start later than in the reference schedule
        to the total number of tasks.
        """
        late_count = 0
        count = 0

        runtime = state.runtime
        for task_id in runtime.completed_tasks:
            if task_id not in self.start_times:
                continue

            ref_start = self.start_times[task_id]
            actual_start = runtime.get_start(task_id)

            if actual_start > ref_start:
                late_count += 1

            count += 1

        return late_count / count if count > 0 else 1.0

    # Order reference metrics

    def order_preservation(self, state: ScheduleState) -> float:
        """
        Calculate the order preservation metric based on the reference schedule.
        This metric is the ratio of the number of tasks that maintain their order
        in the reference schedule to the total number of tasks.
        """

        runtime = state.runtime
        actual_times = [
            runtime.get_start(task_id)
            for task_id, _ in self.sorted_start_times
            if task_id in runtime.completed_tasks
        ]

        n = len(actual_times)
        if n < 2:
            return 1.0

        total_pairs = n * (n - 1) // 2
        inversions = _count_inversions(actual_times)

        return (total_pairs - inversions) / total_pairs

    def hamming_accuracy(self, state: ScheduleState) -> float:
        ref_order = [
            task_id
            for task_id, _ in self.sorted_start_times
            if task_id in state.runtime.completed_tasks
        ]

        actual_order = sorted(
            ref_order,
            key=lambda task_id: state.runtime.get_start(task_id),
        )
        # reference order is just the task_ids in sorted_start_times order
        matches = 0
        for ref_task, act_task in zip(ref_order, actual_order):
            if ref_task == act_task:
                matches += 1

        return matches / len(actual_order) if actual_order else 1.0

    def kendall_tau(self, state: ScheduleState) -> float:
        runtime = state.runtime

        actual_times = [
            runtime.get_start(task_id)
            for task_id, _ in self.sorted_start_times
            if task_id in runtime.completed_tasks
        ]
        n = len(actual_times)
        if n < 2:
            return 1.0

        inversions = _count_inversions(actual_times)
        total_pairs = n * (n - 1) // 2

        ties = 0
        if actual_times:
            elem = actual_times[0]
            for i in range(1, n):
                next_elem = actual_times[i]

                if elem == next_elem:
                    ties += 1

                elem = next_elem

        concordant = total_pairs - inversions
        discordant = inversions
        denominator = sqrt(total_pairs * (total_pairs - ties))

        return (
            (concordant - discordant) / denominator if denominator > 0 else 1.0
        )

    # Machine assignment metrics

    def machine_accuracy(self, state: ScheduleState) -> float:
        matches = 0
        count = 0

        runtime = state.runtime
        for task_id in state.runtime.completed_tasks:
            if task_id not in self.assignments:
                continue

            ref_machine = self.assignments[task_id]
            actual_machine = runtime.get_assignment(task_id)

            if ref_machine == actual_machine:
                matches += 1

            count += 1

        return matches / count if count > 0 else 1.0
