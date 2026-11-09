"""Backtracking utils."""

from copy import copy, deepcopy


class GenericTrail:
    """
    Generic trail implementation for constraints and objectives.

    This class is a generic (non-optimized) util for enabling easy backtracking
    capabilities by trailing a copy of the current state.

    A shallow copy is enough, and much cheaper, when the fields are replaced
    instead of mutated, or hold immutable elements, as lists of numbers.

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
    _deep: bool

    def __init__(self, *fields: str, deep: bool = True) -> None:
        """Initialize a generic trail.

        Parameters
        ----------
        *fields: str
            Field names to trail.
            The objects must have an attribute matching the name.

        deep: bool, optional
            Whether to trail deep copies of the fields. Default to True.

        """
        self._fields = fields
        self._deep = deep

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
            value = getattr(obj, field)
            trail.append(deepcopy(value) if self._deep else copy(value))

    def backtrack(self, mark: int, obj: object) -> None:
        """Restore the state given a checkpoint mark."""
        for field, trail in zip(self._fields, self._trail, strict=True):
            setattr(obj, field, trail[mark])
            del trail[mark:]
