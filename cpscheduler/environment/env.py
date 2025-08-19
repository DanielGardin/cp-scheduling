"""
env.py

This module defines the SchedulingEnv class, which is a custom environment for generic
cheduling problems.
It is designed to be modular and extensible, allowing users to define their own scheduling
problems by specifying the machine setup, constraints, objectives, and instances.

The environment is based on the OpenAI Gym interface, making it compatible with various
reinforcement learning libraries. It provides methods for resetting the environment, taking
steps, rendering the environment, and exporting the scheduling model.
"""

from warnings import warn

from typing import Any
from collections.abc import Iterable, Mapping

from mypy_extensions import u8, i64

from ._common import (
    MAX_INT,
    ProcessTimeAllowedTypes,
    InstanceTypes,
    MachineDataTypes,
    TASK_ID,
    TIME,
    InfoType,
    ObsType,
    InstanceConfig,
    EnvSerialization,
)
from .data import SchedulingData
from .tasks import Tasks
from .instructions import (
    Instruction,
    Signal,
    parse_instruction,
    Action,
    ActionType,
    is_single_action,
)
from .schedule_setup import ScheduleSetup, setups
from .constraints import Constraint, constraints
from .objectives import Objective, objectives
from .metrics import Metric

from ._protocols import ImportableMetric
from .utils import convert_to_list


from ._render import Renderer


def prepare_instance(instance: InstanceTypes) -> dict[str, list[Any]]:
    "Prepare the instance data to a standard dictionary format."
    # features = instance.keys() if isinstance(instance, Mapping) else instance.columns

    return {str(feature): convert_to_list(instance[feature]) for feature in instance}


class SchedulingEnv:
    """
    SchedulingEnv is a custom environment for generic scheduling problems. It is designed to be
    modular and extensible, allowing users to define their own scheduling problems by specifying
    the machine setup, constraints, objectives, and instances.

    The environment is based on the OpenAI Gym interface, making it compatible with various
    reinforcement learning libraries. It provides methods for resetting the environment, taking
    steps, rendering the environment, and exporting the scheduling model.

    Attributes:
        machine_setup: ScheduleSetup
            The machine setup for the scheduling problem.

        constraints: Iterable of Constraint
            A list of constraints to be applied to the scheduling problem.

        objective: Objective
            The objective function to be optimized during scheduling.

        render_mode: str
            The rendering mode for visualizing the scheduling process.

        minimize: bool, optional
            Whether to minimize or maximize the objective function. Default depends on the chosen
            objective.

        allow_preemption: bool, optional, default=False
            Whether to allow preemption in the scheduling process. If True, tasks can be
            interrupted and resumed later.

        instance: InstanceTypes, optional
            The instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing task features and their values.

        processing_times: ProcessTimeAllowedTypes, optional
            The processing times for the tasks, it is dependent on the machine setup.
            If not provided, the environment will attempt to infer processing times from the
            instance data.

        job_instance: InstanceTypes, optional
            The job instance data for the scheduling problem. It can be a DataFrame or a dictionary
            containing job features and their values. If None, no job instance is set.

        job_ids: Iterable[int] | str, optional
            The job IDs for the tasks. If None, job IDs are as the default index of job_instance.

        n_parts: int, optional
            The number of parts to split the tasks into. If None, it defaults to 16 if preemption is
            allowed, otherwise 1.
    """

    # Environment static variables
    setup: ScheduleSetup
    constraints: dict[str, Constraint]
    objective: Objective
    data: SchedulingData

    metrics: dict[str, Metric[Any]]

    # Environment dynamic variables
    tasks: Tasks
    schedule: dict[TASK_ID, list[Instruction]]
    current_time: TIME

    def __init__(
        self,
        machine_setup: ScheduleSetup,
        constraints: Iterable[Constraint] | None = None,
        objective: Objective | None = None,
        instance_config: InstanceConfig | None = None,
        metrics: Mapping[str, Metric[Any]] | None = None,
        render_mode: Renderer | str | None = None,
        allow_preemption: bool = False,
    ):
        self.loaded = False
        self.force_reset = True

        self.preemptive = allow_preemption
        self.setup = machine_setup

        self.constraints = {}
        if constraints is not None:
            for constraint in constraints:
                self.add_constraint(constraint)

        if objective is None:
            objective = Objective()

        self.metrics = {}
        if metrics is not None:
            for name, metric in metrics.items():
                self.add_metric(name, metric)

        self.set_objective(objective)

        self.tasks = Tasks(allow_preemption)
        self.data = SchedulingData()

        self.schedule = {-1: []}

        self.current_time = 0
        self.advancing_to = 0
        self.query_times: list[TIME] = []

        self.renderer = (
            render_mode
            if isinstance(render_mode, Renderer) else
            Renderer.get_renderer(render_mode)
        )

        if instance_config is not None:
            self.set_instance(
                instance=instance_config.get("instance", {}),
                processing_times=instance_config.get("processing_times", None),
                job_instance=instance_config.get("job_instance", None),
                job_feature=instance_config.get("job_feature", ""),
                machine_instance=instance_config.get("machine_instance", None),
            )

    def __repr__(self) -> str:
        if self.loaded:
            return (
                f"SchedulingEnv({self.get_entry()}, n_tasks={self.tasks.n_tasks}, "
                f"current_time={self.current_time}, objective={self._get_objective()})"
            )

        return f"SchedulingEnv({self.get_entry()}, n_tasks=0)"

    # Environment configuration public methods
    def add_constraint(self, constraint: Constraint, replace: bool = False) -> None:
        "Add a constraint to the environment."
        name = constraint.name

        if name in self.constraints and not replace:
            raise ValueError(
                f"Constraint with name {name} already exists. Please use a different name."
            )

        if self.loaded:
            constraint.import_data(self.data)
            constraint.export_data(self.data)

        self.constraints[name] = constraint

    def set_objective(self, objective: Objective) -> None:
        "Set the objective function for the environment."
        if self.loaded:
            objective.import_data(self.data)
            objective.export_data(self.data)

        self.objective = objective

    def add_metric(self, name: str, metric: Metric[Any]) -> None:
        "Add a metric to the environment."
        self.metrics[name] = metric

    def set_instance(
        self,
        instance: InstanceTypes,
        processing_times: ProcessTimeAllowedTypes = None,
        job_instance: InstanceTypes | None = None,
        job_feature: str = "",
        machine_instance: MachineDataTypes | None = None,
    ) -> None:
        """
        Set the instance data for the environment.

        Parameters:
            instance: InstanceTypes
                The instance data for the scheduling problem, can be a DataFrame or a dictionary.

            processing_times: ProcessTimeAllowedTypes
                The processing times for the tasks, can be a list of dictionaries, a list of lists,
                a list of integers, or a string representing a column in the instance data.

            job_instance: Optional[InstanceTypes]
                The job instance data for the scheduling problem, can be a DataFrame or a
                dictionary. If None, no job instance is set.

            job_ids: Optional[Iterable[int] | str]
                The job IDs for the tasks. If None, job IDs are as the default index of the
                job_instance.

            n_parts: Optional[int]
                The number of parts to split the tasks into. If None, it defaults to 16 if
                preemption is allowed, otherwise 1.
        """
        self.data.clear()
        self.tasks.clear()

        task_data = prepare_instance(instance)
        job_data = prepare_instance(job_instance) if job_instance is not None else {}
        machine_data = (
            prepare_instance(machine_instance) if machine_instance is not None else {}
        )

        parsed_processing_times = self.setup.parse_process_time(
            task_data, processing_times
        )

        self.data.add_task_data(parsed_processing_times, task_data, job_feature)
        self.data.add_job_data(job_data)
        self.data.add_machine_data(machine_data)

        for constraint in self.setup.setup_constraints(self.data):
            constraint.setup_constraint = True
            self.add_constraint(constraint, replace=True)

        for constraint in self.constraints.values():
            constraint.import_data(self.data)

            if not constraint.setup_constraint:
                constraint.export_data(self.data)

        self.objective.import_data(self.data)
        self.objective.export_data(self.data)

        for metric in self.metrics.values():
            if isinstance(metric, ImportableMetric):
                metric.import_data(self.data)

        self.tasks.add_tasks(self.data)
        self.loaded = True
        self.force_reset = True

    def get_entry(self) -> str:
        "Get a string representation of the environment's configuration."
        alpha = self.setup.get_entry()

        beta = ",".join(
            [
                constraint.get_entry()
                for constraint in self.constraints.values()
                if not constraint.setup_constraint and constraint.get_entry()
            ]
        )

        if self.preemptive:
            beta += f"{',' if beta else ''}prmp"

        gamma = self.objective.get_entry()

        return f"{alpha}|{beta}|{gamma}"

    # Environment state retrieval methods
    def _get_state(self) -> ObsType:
        "Retrieve the current state of the environment from tasks."
        task_data, job_data = self.data.export_state()
        dynamic_task_data, dynamic_job_data = self.tasks.export_state(self.current_time)

        task_data.update(dynamic_task_data)
        job_data.update(dynamic_job_data)

        return task_data, job_data

    def _get_info(self) -> InfoType:
        "Retrieve additional information about the environment."
        objective_value = self._get_objective()

        info = {
            "n_queries": len(self.query_times),
            "current_time": int(self.current_time),
        }

        for metric_name, metric in self.metrics.items():
            metric_value = metric(
                self.current_time, self.tasks, self.data, objective_value
            )

            if isinstance(metric_value, Mapping):
                info.update(metric_value)

            else:
                info[metric_name] = metric_value

        return info

    def _propagate(self) -> None:
        "Propagate the new bounds through the constraints"
        # Currently not sure if this is needed, but keeping it for future reference.

        # for task_id in self.tasks.awaiting_tasks:
        #     task = self.tasks[task_id]
        #     if task.get_start_lb() < self.current_time:
        #         task.set_start_lb(self.current_time)

        for constraint in self.constraints.values():
            constraint.propagate(self.current_time, self.tasks)

        self.tasks.finish_propagation()

    # Environment API methods
    def reset(
        self, *, options: dict[str, Any] | InstanceConfig | None = None
    ) -> tuple[ObsType, InfoType]:
        if options is not None:
            self.set_instance(
                instance=options.get("instance", {}),
                processing_times=options.get("processing_times", None),
                job_instance=options.get("job_instance", None),
                job_feature=options.get("job_feature", ""),
                machine_instance=options.get("machine_instance", None),
            )

        if not self.loaded:
            raise ValueError("Environment not loaded. Please set an instance first.")

        self.force_reset = False

        self.schedule.clear()
        self.schedule[-1] = []

        self.current_time = 0
        self.advancing_to = 0
        self.query_times.clear()

        self.tasks.reset(self.data)
        for constraint in self.constraints.values():
            constraint.reset(self.tasks)

        self._propagate()

        return self._get_state(), self._get_info()

    def step(
        self,
        action: ActionType = None,
    ) -> tuple[ObsType, float, bool, bool, InfoType]:
        if self.force_reset or not self.loaded:
            raise RuntimeError(
                "Environment was not reset after loading an instance, or wasn't loaded. "
                "Please call reset() after set_instance(...)."
            )

        if is_single_action(action):
            single_args = tuple(map(int, action[1:]))
            self._schedule_instruction(action[0], single_args)

        elif action is not None:
            for instruction in action:
                args = tuple(map(int, instruction[1:]))
                self._schedule_instruction(instruction[0], args)

        self.query_times.append(self.current_time)
        previous_objective = self._get_objective()

        while True:
            if self._dispatch_instruction():
                break

            self.render()

        obs = self._get_state()

        reward = self._get_objective() - previous_objective
        if self.objective.minimize:
            reward = -reward

        truncated = False
        terminal = self._is_terminal()
        info = self._get_info()

        return obs, reward, terminal, truncated, info

    def render(self) -> None:
        self.renderer.render(self.current_time, self.tasks, self.data)

    def _get_objective(self) -> float:
        "Get the current value of the objective function."
        return float(self.objective.get_current(self.current_time, self.tasks))

    def _is_terminal(self) -> bool:
        "Check if the environment is in a terminal state."
        if self.tasks.awaiting_tasks:
            return False

        return all(
            self.tasks[task_id].is_completed(self.current_time)
            for task_id in self.tasks.fixed_tasks
        )

    def _is_schedule_empty(self) -> bool:
        return not self.schedule[-1] and len(self.schedule) == 1

    def _next_decision_time(self, strict: bool = False) -> TIME:
        "Obtain the next decision time to advance. If strict, only consider future tasks."
        next_time = MAX_INT

        if strict:
            for task_id in self.tasks.awaiting_tasks:
                task = self.tasks[task_id]
                start_lb = task.get_start_lb()

                if self.current_time < start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time < instruction_time < next_time:
                    next_time = instruction_time

        else:
            for task_id in self.tasks.awaiting_tasks:
                task = self.tasks[task_id]
                start_lb = task.get_start_lb()
                if start_lb < next_time:
                    next_time = start_lb

            for instruction_time in self.schedule:
                if self.current_time <= instruction_time < next_time:
                    next_time = instruction_time

        if self.current_time < self.advancing_to < next_time:
            next_time = self.advancing_to

        if next_time == MAX_INT:
            return self.tasks.get_time_ub()

        if next_time < self.current_time:
            next_time = self.current_time

        return next_time

    def _advance_to_next_instruction(self) -> None:
        "Advance the environment to the next instruction time."
        next_time = self.advancing_to
        for instruction_time in self.schedule:
            if self.current_time <= instruction_time < next_time:
                next_time = instruction_time

        self.current_time = next_time

    def _advance_to_decision_point(self, strict: bool) -> None:
        "Advance the environment to the next decision point."
        next_time = self._next_decision_time(strict)
        self.current_time = next_time

    def _schedule_instruction(
        self, action: str | Instruction, args: tuple[int, ...]
    ) -> None:
        "Add a single instruction to the schedule."
        instruction, time = parse_instruction(action, args, self.tasks)

        if 0 <= time < self.current_time:
            warn(
                f"Scheduled instruction {instruction} with arguments {args} is in the past. \
                It will be executed immediately."
            )
            time = self.current_time

        if time not in self.schedule:
            self.schedule[time] = []

        self.schedule[time].append(instruction)

    def _process_next_instruction(
        self, scheduled_time: TIME = -1, allow_wait: bool = True
    ) -> bool:
        """
        Process the next instruction in the schedule at the given time.
        If time is -1, use the current time and if allow_wait is False,
        it halts the environment when an instruction is requested to be waited.
        """
        instruction = Instruction()
        signal = Signal(Action.SKIPPED)

        i: i64 = -1
        schedule = self.schedule[scheduled_time]

        while signal.action & Action.SKIPPED and i + 1 < len(schedule):
            i += 1
            instruction = schedule[i]

            signal = instruction.process(self.current_time, self.tasks, self.schedule)

        action: u8 = signal.action
        if not allow_wait and action == Action.WAIT:
            warn(f"{instruction} is not allowed to wait at {self.current_time}.")
            schedule.pop(i)
            return True

        if action & Action.SKIPPED:
            if any(
                self.tasks[task_id].is_executing(self.current_time)
                for task_id in self.tasks.fixed_tasks
            ):
                action = Action.WAIT

            else:
                action = Action.HALT

        if not (action & Action.SKIPPED) and i != -1:
            schedule.pop(i)

        if action & Action.REEVALUATE:
            for task_id in self.tasks.awaiting_tasks:
                self.tasks[task_id].set_start_lb(self.current_time)

        if action & Action.PROPAGATE:
            self._propagate()

        if action & Action.ADVANCE:
            if signal.time > 0:
                self.advancing_to = signal.time
                self._advance_to_next_instruction()

            else:
                self._advance_to_decision_point(strict=False)

        if action & Action.ADVANCE_NEXT:
            self._advance_to_decision_point(strict=True)

        if action & Action.RAISE:
            warn(
                f"Error in instruction {instruction} at time {self.current_time}: {signal.info}"
            )

        return bool(action & Action.HALT)

    def _dispatch_instruction(self) -> bool:
        "Dispatch the next instruction in the schedule depending on the current state."
        if self.current_time in self.schedule:
            halt = self._process_next_instruction(self.current_time, False)

            if not self.schedule[self.current_time]:
                self.schedule.pop(self.current_time)

            return halt

        if self.current_time < self.advancing_to:
            self._advance_to_next_instruction()
            return False

        if self._is_schedule_empty():
            if not self.tasks.awaiting_tasks:
                self.current_time = self.tasks.get_time_ub()

            return True

        return self._process_next_instruction()

    # Custom serialization methods for pickling and deep copying
    def __reduce__(self) -> Any:
        """
        Custom reduce method to ensure the environment can be pickled and deep copied correctly.
        This is necessary for compatibility with multiprocessing and other serialization
        mechanisms.
        """
        return (
            self.__class__,
            (
                self.setup,
                None,
                None,
                None,  # instance_config will be set later
                self.metrics,
                self.renderer,
                self.preemptive,
            ),
            (
                self.constraints,
                self.objective,
                self.data,
                self.tasks,
                self.schedule,
                self.current_time,
                self.advancing_to,
                self.query_times,
                self.loaded,
                self.force_reset,
            ),
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        """
        Custom setstate method to restore the environment's state after unpickling.
        This is necessary to ensure the environment is correctly initialized with its
        data and tasks.
        """
        constraints: dict[str, Constraint]
        (
            constraints,
            self.objective,
            self.data,
            self.tasks,
            self.schedule,
            self.current_time,
            self.advancing_to,
            self.query_times,
            self.loaded,
            self.force_reset,
        ) = state

        for name, constraint in constraints.items():
            if self.loaded:
                constraint.import_data(self.data)
                constraint.refresh(self.current_time, self.tasks)

            self.constraints[name] = constraint

        if self.loaded:
            self.objective.import_data(self.data)

    def to_dict(self, export_data: bool = False) -> EnvSerialization:
        """
        Serialize the environment to a dictionary.

        The resulting serialization is lazily generated, incorporating custom parameters
        for the setup, constraints, and objective into the instance, whenever it's possible.

        For example, if a ReleaseDateConstraint explicitly gets a release time list, instead
        of a tag, the serialization will include that list to the instance, instead of storing
        it on the constraint itself.
        """

        setup_dict = self.setup.to_dict()
        setup_dict["setup"] = self.setup.__class__.__name__

        objective_dict = self.objective.to_dict()
        objective_dict["objective"] = self.objective.__class__.__name__

        constraint_dict: dict[str, dict[str, Any]] = {}
        for constraint in self.constraints.values():
            if constraint.setup_constraint:
                continue

            cls_name = constraint.__class__.__name__
            constraint_dict[cls_name] = constraint.to_dict()

        serialization_dict: EnvSerialization = {
            "setup": setup_dict,
            "constraints": constraint_dict,
            "objective": objective_dict,
        }

        if export_data:
            serialization_dict["instance"] = self.data.to_dict()

        return serialization_dict

    @classmethod
    def from_dict(cls, data: EnvSerialization) -> "SchedulingEnv":
        "Deserialize the environment from a dictionary."
        setup_class = setups[data["setup"].pop("setup")]
        setup = setup_class.from_dict(data["setup"])

        objective_class = objectives[data["objective"].pop("objective")]
        objective = objective_class.from_dict(data["objective"])

        constraints_list = [
            constraints[cls_name].from_dict(constraint_data)
            for cls_name, constraint_data in data["constraints"].items()
        ]

        instance_data = data["instance"] if "instance" in data else None

        return SchedulingEnv(
            machine_setup=setup,
            constraints=constraints_list,
            objective=objective,
            instance_config=instance_data,
        )
