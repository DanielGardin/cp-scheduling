from __future__ import annotations

from typing import Literal, Optional, Iterable, Final, Any, Sequence
from numpy.typing import NDArray

import numpy as np
import numpy.lib.recfunctions as rf

Scalar = int | float | complex | str | np.generic
Selector = int | slice | NDArray[np.integer[Any]] | NDArray[np.bool] | Sequence[int]

MIN_INT: Final[int] = -2 ** 31 + 1
MAX_INT: Final[int] =  2 ** 31 - 1

class _Bound:
    def __init__(self, array: NDArray[np.int32], modifier: int | NDArray[np.int32] = 0) -> None:
        self._array    = array
        self._modifier = modifier

    def __repr__(self) -> str:
        return repr(self._array + self._modifier)

    def __setitem__(self, indices: Selector, value: NDArray[np.int32] | int) -> None:
        if isinstance(self._modifier, int):
            self._array[indices] = value - self._modifier
            return

        self._array[indices] = value - self._modifier[indices]


    def __getitem__(self, indices: Selector) -> NDArray[np.int32]:
        if isinstance(self._modifier, int):
            return self._array[indices] + self._modifier

        return self._array[indices] + self._modifier[indices]



def export_single_variable(name: str, size: int, lb: int, ub: int) -> str:
    rep = f"{name} = intervalVar(size={size});\n"

    if lb == ub:
        rep += f"startOf({name}) == {lb};"
    
    else:
        rep += f"startOf({name}) >= {lb};\n"
        rep += f"startOf({name}) <= {ub};"

    return rep


class IntervalVars:
    tasks:     NDArray[np.void]
    durations: NDArray[np.int32]
    _start_lb:  NDArray[np.int32]
    _start_ub:  NDArray[np.int32]

    ids: NDArray[np.generic]
    _indices: dict[Scalar, int]

    to_propagate: NDArray[np.bool]

    NAME= r"IntervalVars_$1"

    def __init__(
            self,
            tasks: NDArray[np.void],
            durations: NDArray[np.int32],
            task_ids: Optional[Iterable[Scalar]] = None,
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
        
        task_ids: Optional[Iterable[Scalar]], default=None
            The ids of the tasks. If not provided, the ids will be the range of the number of tasks.
        """


        self.features = tasks
        self.durations = np.asarray(durations, dtype=np.int32)

        self._start_lb = np.zeros(len(tasks), dtype=np.int32)
        self._start_ub = np.full(len(tasks), MAX_INT, dtype=np.int32)

        if task_ids is None:
            task_ids = range(len(tasks))

        self._indices = {task_id: i for i, task_id in enumerate(task_ids)}
        self.ids  = np.asarray(list(self._indices.keys()))

        self.to_propagate = np.ones(len(tasks), dtype=np.bool)


    def __len__(self) -> int:
        return len(self.features)

    @property
    def dtype(self) -> np.dtype[np.void]:
        return self.features.dtype


    @property
    def index_dtype(self) -> np.dtype[np.generic]:
        return self.ids.dtype


    @property
    def names(self) -> list[str]:
        return [self.NAME.replace("$1", str(task_id)) for task_id in self._indices]

    @property
    def start_lb(self) -> _Bound:
        return _Bound(self._start_lb)
    
    @property
    def start_ub(self) -> _Bound:
        return _Bound(self._start_ub)

    @property
    def end_lb(self) -> _Bound:
        return _Bound(self._start_lb, self.durations)

    @property
    def end_ub(self) -> _Bound:
        return _Bound(self._start_ub, self.durations)


    def __getitem__(self, feature: str) -> NDArray[Any]:
        return self.features[feature]


    def fix_start(self, tasks: Scalar | Iterable[Scalar], value: int) -> None:
        indices = self.get_indices(tasks)

        self.start_lb[indices] = value
        self.start_ub[indices] = value

        self.to_propagate[indices] = True


    def is_fixed(self) -> NDArray[np.bool]:
        fixed: NDArray[np.bool] = self.start_lb[:] == self.start_ub[:]

        return fixed


    def is_awaiting(self) -> NDArray[np.bool]:
        return ~self.is_fixed()   


    def is_executing(self, time: int) -> NDArray[np.bool]:
        return self.is_fixed() & \
            (self.start_lb[:] <= time) & \
            (time < self.end_lb[:])


    def is_finished(self, time: int) -> NDArray[np.bool]:
        return self.is_fixed() & (time >= self.end_lb[:])


    def clear_tasks(self) -> None:
        self.features = np.zeros((0,), dtype=self.features.dtype)
        self.durations = np.zeros((0,), dtype=self.durations.dtype)

        self._start_lb = np.zeros((0,), dtype=self._start_lb.dtype)
        self._start_ub = np.zeros((0,), dtype=self._start_ub.dtype)

        self._indices = {}

        self.to_propagate = np.zeros((0,), dtype=self.to_propagate.dtype)


    def add_tasks(
            self,
            tasks: NDArray[np.void],
            durations: NDArray[np.int32],
            task_ids: Optional[Iterable[Scalar]] = None
        ) -> None:

        if task_ids is None:
            task_ids = range(len(self.features), len(self.features) + len(tasks))

        self._indices.update({task_id: i for i, task_id in enumerate(task_ids)})

        self.features = np.concatenate([self.features, tasks])
        self.durations = np.concatenate([self.durations, durations])

        self._start_lb = np.concatenate([self._start_lb, np.zeros(len(tasks), dtype=np.int32)])
        self._start_ub = np.concatenate([self._start_ub, np.full(len(tasks), MAX_INT, dtype=np.int32)])

        self.to_propagate = np.concatenate([self.to_propagate, np.ones(len(tasks), dtype=np.bool)])


    def reset_state(self) -> None:
        self.start_lb[:] = 0
        self.start_ub[:] = MAX_INT

        self.to_propagate[:] = True


    def get_indices(self, indices: Scalar | Iterable[Scalar]) -> NDArray[np.int32]:
        if isinstance(indices, Scalar):
            return np.array(self._indices[indices], dtype=np.int32)
        
        return np.array([self._indices[index] for index in indices], dtype=np.int32)


    def export_variables(self) -> str:
        names = self.names

        variables = [
            export_single_variable(name, int(duration), int(lb), int(ub)) for name, duration, lb, ub in zip(names, self.durations, self._start_lb, self._start_ub)
        ]

        return '\n'.join(variables)


    def get_state(self, current_time: int) -> NDArray[np.void]:
        buffer = np.zeros(len(self.features), dtype=np.dtypes.StrDType(9))
        remaining_time = np.zeros_like(self.durations)

        is_fixed     = self.is_fixed()
        is_executing = self.is_executing(current_time)

        remaining_time[is_executing] = self.end_lb[is_executing] - current_time
        remaining_time[~is_fixed]    = self.durations[~is_fixed]

        buffer[~is_fixed]    = 'awaiting'
        buffer[is_executing] = 'executing'
        buffer[is_fixed & ~is_executing] = 'finished'


        state = rf.append_fields(
            self.features.copy(),
            ('remaining_time', 'buffer'),
            (remaining_time, buffer),
            usemask=False
        )

        return np.asarray(state)