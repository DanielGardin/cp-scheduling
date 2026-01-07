from copy import deepcopy

from cpscheduler.environment._common import TASK_ID, TIME, MACHINE_ID
from cpscheduler.environment.env import SchedulingEnv
from cpscheduler.environment.instructions import ActionType
from cpscheduler.environment.state import ScheduleState


def machine_utilization(time: int, state: ScheduleState, objective: float) -> float:
    """
    Calculate the percentage of time that machines are utilized during the scheduling period.
    """

    total_time = time
    busy_time = 0
    for task in state.fixed_tasks:
        busy_time += task.get_processed_time(time)

    return state.n_machines * busy_time / total_time if total_time > 0 else 1


def max_preemptions(time: int, state: ScheduleState, objective: float) -> int:
    "Calculate the maximum number of preemption switches that occurred during the scheduling period."
    max_switches = 0
    for task in state.fixed_tasks:
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

    start_times: dict[TASK_ID, TIME]
    assignments: dict[TASK_ID, MACHINE_ID]

    objective_value: float

    def __init__(self, reference_schedule: ActionType, env: SchedulingEnv):
        self.start_times = {}
        self.assignments = {}

        cpy_env = deepcopy(env)
        *_, info = cpy_env.step(reference_schedule)

        for task in cpy_env.state.fixed_tasks:
            self.start_times[task.task_id] = task.get_start()
            self.assignments[task.task_id] = task.get_assignment()

        self.objective_value = info["objective_value"]

    def __call__(
        self, time: int, state: ScheduleState, objective: float
    ) -> dict[str, float]:
        self.force_import = False

        metrics = {
            "mean_displacement_distance": self.mean_displacement_distance(
                time, state, objective
            ),
            "order_preservation": self.order_preservation(time, state, objective),
            "hamming_accuracy": self.hamming_accuracy(time, state, objective),
            "kendall_tau": self.kendall_tau(time, state, objective),
        }

        return metrics

    def mean_displacement_distance(
        self, time: int, state: ScheduleState, objective: float
    ) -> float:
        """
        Calculate the mean displacement distance of the reference schedule.
        The displacement distance is the sum of the absolute differences between
        the scheduled and actual start times of each task.
        """
        distance = 0
        count = 0

        for task_id, reference_time in self.start_times.items():
            task = state.tasks[task_id]

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
        self, time: int, state: ScheduleState, objective: float
    ) -> float:
        """
        Calculate the order preservation metric based on the reference schedule.
        This metric is the ratio of the number of tasks that maintain their order
        in the reference schedule to the total number of tasks.
        """

        start_times = [
            (start_time, state.tasks[task_id].get_start())
            for task_id, start_time in self.start_times.items()
            if state.tasks[task_id].is_fixed()
        ]

        n = len(start_times)
        if n < 2:
            return 1.0

        start_times.sort()

        actual_times = [actual_start for _, actual_start in start_times]

        preserved_count = 0
        total_count = n * (n - 1) // 2

        for i in range(n - 1):
            ai = actual_times[i]

            for j in range(i + 1, n):
                if ai <= actual_times[j]:
                    preserved_count += 1

        return preserved_count / total_count

    # The following are useful when the schedule is a permutation of the tasks
    def _get_permutations(
        self, state: ScheduleState
    ) -> tuple[list[TASK_ID], list[TASK_ID]]:
        actual = sorted(
            [
                (task.get_start(), task.task_id)
                for task in state.fixed_tasks
                if task.task_id in self.start_times
            ]
        )

        reference = sorted(
            [(self.start_times[task_id], task_id) for _, task_id in actual]
        )

        return (
            [task_id for _, task_id in reference],
            [task_id for _, task_id in actual],
        )

    def hamming_accuracy(
        self, time: int, state: ScheduleState, objective: float
    ) -> float:
        """
        Calculate the Hamming similarity of the schedule permutation compared to the reference
        schedule. This metric is the number of tasks that are in the same position
        as in the reference schedule.
        """
        reference_perm, actual_perm = self._get_permutations(state)

        hamming_count = 0
        for ref_task, act_task in zip(reference_perm, actual_perm):
            if ref_task == act_task:
                hamming_count += 1

        return hamming_count / len(reference_perm) if reference_perm else 1.0

    def kendall_tau(self, time: int, state: ScheduleState, objective: float) -> float:
        """
        Calculate the Kendall Tau distance between the reference schedule and the actual schedule.
        This metric is the number of discordant pairs in the schedule.
        """
        reference_perm, actual_perm = self._get_permutations(state)
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

                if ref_i == ref_j:
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

    def regret(self, time: int, state: ScheduleState, objective: float) -> float:
        """
        Calculate the regret of the current schedule compared to the reference schedule.
        Regret is defined as the difference between the reference objective value
        and the current objective value.
        """
        return self.objective_value - objective

    def performance_gap(
        self, time: int, state: ScheduleState, objective: float
    ) -> float:
        """
        Calculate the performance gap of the current schedule compared to the reference schedule.
        The performance gap is defined as the ratio of the regret to the reference objective value.
        """
        return objective / (self.objective_value + 1e-9) - 1.0
