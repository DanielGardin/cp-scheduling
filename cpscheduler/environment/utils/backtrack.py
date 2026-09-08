"""Backtracking utils."""

from copy import deepcopy


class GenericTrail:
    """
    Generic trail implementation for constraints and objectives.

    This class is a generic (non-optimized) util for enabling easy backtracking
    capabilities by trailing a (deep)copy of the current state.

    Example
    -------
    >>> class MyObjective(Objective):
    ...     value: int
    ...     lb: int
    ...     static_expensive_list: list[int]
    ...
    ...     def __init__(self, param: list[int]):
    ...         self.static_expensive_list = param
    ...         self.value = int
    ...         self.lb = int
    ...         self._trail = GenericTrail("value", "lb")
    ...
    ...     def checkpoint(self, mark: int) -> None:
    ...         self._trail.checkpoint(mark, self)
    ...
    ...     def backtrack(self, mark: int, changed_tasks: set[TaskID], state: ScheduleState) -> None:
    ...         self._trail.backtrack(mark, self)

    """

    _fields: tuple[str, ...]
    _trail: list[list[object]]

    def __init__(self, *fields: str) -> None:
        """Initialize a generic trail.

        Parameters
        ----------
        *fields: str
            Field names to trail.
            The objects must have an attribute matching the name.

        """
        self._fields = fields

        self._trail = [[] for _ in fields]

    def reset(self) -> None:
        """Reset the trail."""
        for trail in self._trail:
            trail.clear()

    def checkpoint(self, mark: int, obj: object) -> None:
        """Checkpoints the current state for backtracking."""
        if mark == 0:
            self.reset()

        assert all(len(trail) == mark for trail in self._trail)

        for field, trail in zip(self._fields, self._trail, strict=True):
            trail.append(deepcopy(getattr(obj, field)))

    def backtrack(self, mark: int, obj: object) -> None:
        """Restore the state given a checkpoint mark."""
        for field, trail in zip(self._fields, self._trail, strict=True):
            setattr(obj, field, trail[mark])
            del trail[mark:]
