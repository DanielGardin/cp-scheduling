from typing import Any, Optional, overload, Iterable

from .utils import MAX_INT, AVAILABLE_SOLVERS, invert, convert_to_list

def export_single_variable_cplex(name: str, size: int, lb: int, ub: int) -> str:
    rep = f"{name} = intervalVar(size={size});\n"

    if lb == ub:
        rep += f"startOf({name}) == {lb};"

    else:
        rep += f"startOf({name}) >= {lb};\n"
        rep += f"startOf({name}) <= {ub};"

    return rep


def export_single_variable_ortools(name: str, size: int, lb: int, ub: int) -> str:
    rep = f"{name}_start = model.NewIntVar({lb}, {ub}, '{name}_start')\n"

    end_ub = MAX_INT if ub == MAX_INT else ub + size
    rep += f"{name}_end = model.NewIntVar({lb + size}, {end_ub}, '{name}_end')\n"

    rep += f"{name} = model.NewIntervalVar({name}_start, {size}, {name}_end, '{name}')"

    return rep


class IntervalVars:
    tasks:      dict[str, list[Any]]
    durations:  list[int]
    _start_lb:  list[int]
    _start_ub:  list[int]

    _to_propagate: list[bool]

    NAME= r"IntervalVars"

    def __init__(
            self,
            tasks: dict[str, list[Any]],
            durations: list[int],
        ) -> None:
        """
        Initialize the IntervalVars object. Each task is represented by a numpy void type and
        are considered as the features of the object and take a deterministic and continuous slot
        in the timeline.

        Use this variable for non-preemptive tasks.

        Parameters:
        ----------
        tasks: NDArray[np.void], shape=(n_tasks,)
            The tasks to be scheduled. Each task is a tuple of features.

        durations: NDArray[np.int32], shape=(n_tasks,)
            The duration of each task.
        """
        self.durations = durations

        if any([len(feat) != len(self.durations) for feat in tasks.values()]):
            raise ValueError("All features must have the same length as the number of tasks.")

        self.features = tasks

        self.n_tasks = len(self.durations)

        self._start_lb     = [0]       * self.n_tasks
        self._start_ub     = [MAX_INT] * self.n_tasks
        self._to_propagate = [True]    * self.n_tasks

    def __len__(self) -> int:
        return self.n_tasks


    @overload
    def get_var_name(self, tasks: int) -> str:
        ...

    @overload
    def get_var_name(self, tasks: Optional[Iterable[int]] = None) -> list[str]:
        ...


    def get_var_name(self, tasks: Optional[int | Iterable[int]] = None) -> str | list[str]:
        if isinstance(tasks, int):
            return f"{self.NAME}_{tasks}"

        if tasks is None:
            tasks = range(self.n_tasks)

        return [f"{self.NAME}_{task_id}" for task_id in tasks]


    def __getitem__(self, feature: str) -> list[Any]:
        return self.features[feature]


    @overload
    def get_features(self, tasks: int) -> dict[str, Any]:
        ...

    @overload
    def get_features(self, tasks: Optional[Iterable[int]] = None) -> dict[str, list[Any]]:
        ...


    def get_features(self, tasks: Optional[int | Iterable[int]] = None) -> dict[str, Any | list[Any]]:
        if isinstance(tasks, int):
            return {feature: self.features[feature][tasks] for feature in self.features}

        if tasks is None:
            tasks = range(self.n_tasks)

        return {feature: [self.features[feature][task] for task in tasks] for feature in self.features}


    def fix_start(self, tasks: int | Iterable[int], value: int) -> None:
        if isinstance(tasks, int):
            tasks = [tasks]

        for task in tasks:
            self._start_lb[task]     = value
            self._start_ub[task]     = value
            self._to_propagate[task] = True


    def fix_end(self, tasks: int | Iterable[int], value: int) -> None:
        indices = tasks

        if isinstance(indices, int):
            indices = [indices]

        for task in indices:
            self._start_lb[task] = value - self.durations[task]
            self._start_ub[task] = value - self.durations[task]
            self._to_propagate[task] = True


    def set_start_lb(self, tasks: int | Iterable[int], value: int) -> None:
        if isinstance(tasks, int):
            tasks = [tasks]

        for task in tasks:
            if value > self._start_lb[task] and not self.is_fixed(task):
                self._start_lb[task]     = value
                self._to_propagate[task] = True


    def set_start_ub(self, tasks: int | Iterable[int], value: int) -> None:
        if isinstance(tasks, int):
            tasks = [tasks]

        for task in tasks:
            if value < self._start_ub[task] and not self.is_fixed(task):
                self._start_ub[task]     = value
                self._to_propagate[task] = True


    def set_end_lb(self, tasks: int | Iterable[int], value: int) -> None:
        if isinstance(tasks, int):
            tasks = [tasks]

        for task in tasks:
            if value > self._start_lb[task] and not self.is_fixed(task):
                self._start_lb[task]    = value - self.durations[task]
                self._to_propagate[task] = True


    def set_end_ub(self, tasks: int | Iterable[int], value: int) -> None:
        if isinstance(tasks, int):
            tasks = [tasks]

        for task in tasks:
            if value < self._start_ub[task] and not self.is_fixed(task):
                self._start_lb[task]    = value - self.durations[task]
                self._to_propagate[task] = True


    @overload
    def get_duration(self, tasks: int) -> int:
        ...

    @overload
    def get_duration(self, tasks: Optional[Iterable[int]] = None) -> list[int]:
        ...

    def get_duration(self, tasks: Optional[int | Iterable[int]] = None) -> int | list[int]:
        if isinstance(tasks, int):
            return self.durations[tasks]

        if tasks is None:
            tasks = range(self.n_tasks)

        return [self.durations[task] for task in tasks]


    @overload
    def get_start_lb(self, tasks: int) -> int:
        ...

    @overload
    def get_start_lb(self, tasks: Optional[Iterable[int]] = None) -> list[int]:
        ...

    def get_start_lb(self, tasks: Optional[int | Iterable[int]] = None) -> int | list[int]:
        if isinstance(tasks, int):
            return self._start_lb[tasks]

        if tasks is None:
            tasks = range(self.n_tasks)

        return [self._start_lb[task] for task in tasks]


    @overload
    def get_start_ub(self, tasks: int) -> int:
        ...

    @overload
    def get_start_ub(self, tasks: Optional[Iterable[int]] = None) -> list[int]:
        ...

    def get_start_ub(self, tasks: Optional[int | Iterable[int]] = None) -> int | list[int]:
        if isinstance(tasks, int):
            return self._start_ub[tasks]

        if tasks is None:
            tasks = range(self.n_tasks)

        return [self._start_ub[task] for task in tasks]


    @overload
    def get_end_lb(self, tasks: int) -> int:
        ...

    @overload
    def get_end_lb(self, tasks: Optional[Iterable[int]] = None) -> list[int]:
        ...

    def get_end_lb(self, tasks: Optional[int | Iterable[int]] = None) -> int | list[int]:
        if isinstance(tasks, int):
            return min(self._start_lb[tasks] + self.durations[tasks], MAX_INT)

        if tasks is None:
            tasks = range(self.n_tasks)

        return [min(self._start_lb[index] + self.durations[index], MAX_INT) for index in tasks]


    @overload
    def get_end_ub(self, tasks: int) -> int:
        ...

    @overload
    def get_end_ub(self, tasks: Optional[Iterable[int]] = None) -> list[int]:
        ...

    def get_end_ub(self, tasks: Optional[int | Iterable[int]] = None) -> int | list[int]:
        if isinstance(tasks, int):
            return min(self._start_ub[tasks] + self.durations[tasks], MAX_INT)

        if tasks is None:
            tasks = range(self.n_tasks)

        return [min(self._start_ub[index] + self.durations[index], MAX_INT) for index in tasks]


    @overload
    def is_fixed(self, tasks: int) -> bool:
        ...

    @overload
    def is_fixed(self, tasks: Optional[Iterable[int]] = None) -> list[bool]:
        ...

    def is_fixed(self, tasks: Optional[int | Iterable[int]] = None) -> bool | list[bool]:
        if isinstance(tasks, int):
            return self._start_lb[tasks] == self._start_ub[tasks]

        if tasks is None:
            tasks = range(self.n_tasks)

        return [self._start_lb[index] == self._start_ub[index] for index in tasks]


    @overload
    def is_awaiting(self, tasks: int, time: Optional[int] = None) -> bool:
        ...

    @overload
    def is_awaiting(self, tasks: Iterable[int], time: Optional[int] = None) -> list[bool]:
        ...

    def is_awaiting(self, tasks: int | Iterable[int], time: Optional[int] = None) -> bool | list[bool]:
        if time is None:
            return invert(self.is_fixed(tasks))

        if isinstance(tasks, int):
            return not self.is_fixed(tasks) or time < self.get_start_lb(tasks)

        is_fixed_list = self.is_fixed(tasks)
        starts        = self.get_start_lb(tasks)

        return [not is_fixed or time < start for is_fixed, start in zip(is_fixed_list, starts)]


    @overload
    def is_executing(self, tasks: int, time: int) -> bool:
        ...

    @overload
    def is_executing(self, tasks: Iterable[int], time: int) -> list[bool]:
        ...

    def is_executing(self, tasks: int | Iterable[int], time: int) -> bool | list[bool]:
        if isinstance(tasks, int):
            return self.is_fixed(tasks) and self.get_start_lb(tasks) <= time and (time < self.get_end_lb(tasks))

        is_fixed_list = self.is_fixed(tasks)
        starts        = self.get_start_lb(tasks)
        ends          = self.get_end_lb(tasks)

        return [is_fixed and start <= time and time < end for is_fixed, start, end in zip(is_fixed_list, starts, ends)]


    @overload
    def is_finished(self, tasks: int, time: int) -> bool:
        ...

    @overload
    def is_finished(self, tasks: Iterable[int], time: int) -> list[bool]:
        ...

    def is_finished(self, tasks: int | Iterable[int], time: int) -> bool | list[bool]:
        if isinstance(tasks, int):
            return self.is_fixed(tasks) and self.get_end_lb(tasks) <= time

        is_fixed_list = self.is_fixed(tasks)
        ends          = self.get_end_lb(tasks)

        return [is_fixed and end < time for is_fixed, end in zip(is_fixed_list, ends)]


    @overload
    def is_available(self, tasks: int, time: int) -> bool:
        ...

    @overload
    def is_available(self, tasks: Iterable[int], time: int) -> list[bool]:
        ...

    def is_available(self, tasks: int | Iterable[int], time: int) -> bool | list[bool]:
        if isinstance(tasks, int):
            return not self.is_fixed(tasks) and self.get_start_lb(tasks) <= time

        is_fixed  = self.is_fixed(tasks)
        start_lbs = self.get_start_lb(tasks)

        return [not is_fixed and start_lb <= time for is_fixed, start_lb in zip(is_fixed, start_lbs)]


    @overload
    def to_propagate(self, tasks: int) -> bool:
        ...

    @overload
    def to_propagate(self, tasks: Optional[Iterable[int]] = None) -> list[bool]:
        ...

    def to_propagate(self, tasks: Optional[int | Iterable[int]] = None) -> bool | list[bool]:
        if isinstance(tasks, int):
            return self._to_propagate[tasks]

        if tasks is None:
            tasks = range(self.n_tasks)

        return [self._to_propagate[index] for index in tasks]


    def add_tasks(
            self,
            tasks: dict[str, list[int]],
            durations: Iterable[int],
            task_ids: Optional[Iterable[int]] = None
        ) -> None:
        for feature in tasks:
            self.features[feature].extend(tasks[feature])

        self.durations.extend(durations)

        n_new_tasks = len(self.durations) - self.n_tasks

        self._start_lb.extend([0] * n_new_tasks)
        self._start_ub.extend([MAX_INT] * n_new_tasks)
        self._to_propagate.extend([True] * n_new_tasks)

        self.n_tasks = len(self.durations)


    def reset_state(self) -> None:
        for index in range(len(self._start_lb)):
            self._start_lb[index] = 0
            self._start_ub[index] = MAX_INT
            self._to_propagate[index] = True


    def export_variables(self, solver: AVAILABLE_SOLVERS = 'cplex') -> str:
        names = self.get_var_name()

        if solver == 'cplex':
            variables = [
                export_single_variable_cplex(name, duration, lb, ub) for name, duration, lb, ub in zip(names, self.durations, self._start_lb, self._start_ub)
            ]

        else:
            variables = [
                export_single_variable_ortools(name, duration, lb, ub) for name, duration, lb, ub in zip(names, self.durations, self._start_lb, self._start_ub)
            ]


        return '\n'.join(variables)


    def get_buffer(self, task: int, current_time: int) -> str:
        if self.is_available(task, current_time):
            return 'available'

        if self.is_awaiting(task, current_time):
            return 'awaiting'

        if self.is_finished(task, current_time):
            return 'finished'

        if self.is_executing(task, current_time):
            return 'executing'

        return 'unknown'


    def get_state(self, current_time: int) -> dict[str, list[Any]]:
        buffer = [self.get_buffer(task, current_time) for task in range(self.n_tasks)]

        remaining_time = [
            min(max(end - current_time, 0), duration)
            for end, duration in zip(self.get_end_lb(), self.durations)
        ]

        return {
            'task_id': convert_to_list(range(self.n_tasks)),
            **self.features,
            'remaining_time': remaining_time,
            'buffer': buffer,
        }