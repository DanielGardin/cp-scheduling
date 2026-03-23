from typing import Any
from typing_extensions import NamedTuple

from cpscheduler.environment.constants import Time

class ViolationRecord(NamedTuple):
    "A record of a constraint violation"

    time: Time
    penalty: float
    constraint: str

class ViolationState:
    """
    Container for the violation state of the scheduling environment.

    This class is used to store any information about constraint violations that may be
    needed by constraints or other components of the environment. It is designed to be
    flexible and can be extended with additional attributes as needed.
    """

    __slots__ = ("violations", "total_penalty")

    violations: dict[str, list[ViolationRecord]]
    total_penalty: float

    def __init__(self) -> None:
        self.violations = {}
        self.total_penalty = 0.0

    def record_violation(self, time: Time, penalty: float, constraint: str) -> None:
        "Record a constraint violation with the given time, penalty, and constraint name."
        if constraint not in self.violations:
            self.violations[constraint] = []

        self.violations[constraint].append(
            ViolationRecord(time, penalty, constraint)
        )
        self.total_penalty += penalty

    def clear_violations(self) -> None:
        "Clear all recorded violations and reset the total penalty."
        self.violations.clear()
        self.total_penalty = 0.0
    
    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (), (self.violations, self.total_penalty))
    
    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.violations, self.total_penalty = state
    
    def get_violations(self, constraint: str) -> list[ViolationRecord]:
        "Get the list of violations for a specific constraint."
        return self.violations.get(constraint, [])

