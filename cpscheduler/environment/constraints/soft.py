from cpscheduler.environment.utils import convert_to_list

from cpscheduler.environment.constants import Time
from cpscheduler.environment.state import ScheduleState

from cpscheduler.environment.constraints.base import SoftConstraint


class SoftDeadlineConstraint(SoftConstraint, violation_name="deadline"):
    due_tag: str
    due_dates: list[Time]

    def __init__(self, due_dates: str = "due_date") -> None:
        self.due_tag = due_dates
        self.due_dates = []

    def initialize(self, state: ScheduleState) -> None:
        self.due_dates = convert_to_list(
            state.instance.task_instance[self.due_tag], Time
        )

    def on_assignment(
        self, task_id: Time, machine_id: Time, state: ScheduleState
    ) -> None:
        due_date = self.due_dates[task_id]
        completion_time = state.get_end_lb(task_id)

        if completion_time > due_date:
            delay = completion_time - due_date

            state.record_violation(self.violation_name, float(delay))
