"""Metrics utilities for scheduling environments.

This module provides commonly used performance metrics for scheduling
experiments.
These metrics are not exhaustive, any function that takes the current
`ScheduleState` and returns a scalar value can be used as a metric.
"""

from collections import Counter
from copy import deepcopy
from math import sqrt
from typing import Any

from cpscheduler.environment.constants import MachineID, TaskID, Time
from cpscheduler.environment.des import ActionType
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.state import ScheduleState


def machine_utilization(state: ScheduleState) -> float:
    """Calculate the percentage of time that machines are utilized during the scheduling period.

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
    """Calculate the total number of preemption switches that occurred during the scheduling period."""
    total_switches = 0

    for task_id in state.runtime.completed_tasks:
        history = state.runtime.history[task_id]
        total_switches += len(history) - 1

    return total_switches


def max_preemptions(state: ScheduleState) -> int:
    """Calculate the maximum number of preemption switches that occurred during the scheduling period."""
    max_switches = 0
    for task_id in state.runtime.completed_tasks:
        history = state.runtime.history[task_id]
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
    """Compute comparison metrics between a reference and executed schedule.

    The class captures the reference schedule by executing it on a copy of the
    provided `SchedulingEnv` and recording each task's final start time and
    machine assignment (preemption is not modeled in the reference).

    After construction, calling the instance with a `ScheduleState` returns a
    dictionary of numeric metrics comparing the reference against the provided
    state's completed tasks.
    """

    start_times: dict[TaskID, Time]
    assignments: dict[TaskID, MachineID]

    sorted_start_times: list[tuple[TaskID, Time]]

    def __init__(self, env: SchedulingEnv, reference_schedule: ActionType):
        """Record the reference start times and assignments.

        Parameters
        ----------
        env : SchedulingEnv
            Environment used to execute the reference schedule (will be deep-
            copied to avoid modifying the original).

        reference_schedule : ActionType
            A single action or batch representing the reference schedule.

        """
        self.start_times = {}
        self.assignments = {}

        cpy_env = deepcopy(env)
        runtime = cpy_env.state.runtime

        already_completed = runtime.completed_tasks.copy()

        cpy_env.step(reference_schedule)

        for task_id in runtime.completed_tasks - already_completed:
            self.assignments[task_id] = runtime.get_assignment(task_id)
            self.start_times[task_id] = runtime.get_start(task_id)

        self.sorted_start_times = sorted(
            [(task_id, start) for task_id, start in self.start_times.items()],
            key=lambda x: x[1],
        )

    def __call__(self, state: ScheduleState) -> dict[str, float]:
        """Return a dictionary of reference comparison metrics for `state`."""
        return {
            "mean_displacement_distance": self.mean_displacement_distance(
                state
            ),
            "order_preservation": self.order_preservation(state),
            "hamming_accuracy": self.hamming_accuracy(state),
            "kendall_tau": self.kendall_tau(state),
            "machine_accuracy": self.machine_accuracy(state),
        }

    # Time reference metrics

    def mean_displacement_distance(self, state: ScheduleState) -> float:
        """Calculate the mean displacement distance of the reference schedule.

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
        """Calculate the early displacement ratio of the reference schedule.

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
        """Calculate the late displacement ratio of the reference schedule.

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
        """Calculate the order preservation metric based on the reference schedule.

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
        """Compute the Hamming-style accuracy of task positions.

        Returns the fraction of tasks whose position in the executed order
        matches their position in the reference order.
        """
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
        for ref_task, act_task in zip(ref_order, actual_order, strict=False):
            if ref_task == act_task:
                matches += 1

        return matches / len(actual_order) if actual_order else 1.0

    def kendall_tau(self, state: ScheduleState) -> float:
        """Compute Kendall Tau correlation between reference and executed order.

        Returns a normalized Kendall Tau-like score in [-1, 1], with 1.0
        indicating identical order and values closer to -1 indicating
        strong disagreement.
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

        inversions = _count_inversions(actual_times)
        total_pairs = n * (n - 1) // 2

        # Count all tied pairs, not just adjacent equal values.
        ties = sum(
            count * (count - 1) // 2 for count in Counter(actual_times).values()
        )

        concordant = total_pairs - inversions - ties
        discordant = inversions
        denominator = sqrt(total_pairs * (total_pairs - ties))

        return (
            (concordant - discordant) / denominator if denominator > 0 else 1.0
        )

    # Machine assignment metrics

    def machine_accuracy(self, state: ScheduleState) -> float:
        """Return the fraction of completed tasks assigned to the same machine as in the reference schedule."""
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
