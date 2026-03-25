from typing import Any
from collections.abc import Sequence, Iterable, Mapping

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.env import SchedulingEnv


class CPVariables:
    def __init__(self, state: ScheduleState):
        self.n_tasks = state.n_tasks
        self.n_jobs = state.n_jobs
        self.n_machines = state.n_machines

        self.horizon = max(
            state.get_end_ub(task_id) for task_id in state.awaiting_tasks
        )
