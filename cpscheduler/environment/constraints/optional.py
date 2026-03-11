from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import PassiveConstraint


class RejectableConstraint(PassiveConstraint):
    """
    A scheduling problem with rejection allows a subset of tasks to be rejected,
    not contributing to the objective function, but usually incurring a penalty
    cost.
    """

    def initialize(self, state: ScheduleState) -> None:
        for task_id in range(state.n_tasks):
            state.instance.set_optionality(task_id)

    def get_entry(self) -> str:
        return "rej"
