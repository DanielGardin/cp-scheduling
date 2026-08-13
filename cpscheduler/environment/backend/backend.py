"""Backend abstract class for dispatching a schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle, Time

if TYPE_CHECKING:
    from cpscheduler.environment.backend.actions import Instruction
    from cpscheduler.environment.state import ScheduleState


EventID = int

backends: dict[str, type[ScheduleBackend]] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=False)
class ScheduleBackend(EzPickle):
    """Schedule backend used for dispatching tasks.

    This class stores and manages the current schedule.
    """

    backend: ClassVar[str] = ""
    _next_event_id: EventID

    @override
    def __init_subclass__(cls) -> None:

        if not cls.backend:
            raise ValueError(
                f"Backend `{cls.__name__}` does not define `backend` "
                "class variable."
            )

        backends[cls.backend] = cls

    @classmethod
    def from_register(cls, backend: str) -> ScheduleBackend:
        """Return the backend registered by its name."""
        backend_cls = backends.get(backend)

        if backend_cls is None:
            all_backends = ",".join(backends)

            raise ValueError(
                f"No backend {backend} registered, choose one of the following: "
                f"{all_backends}"
            )

        return backend_cls()

    def reset(self) -> None:
        """Reset the schedule to its initial empty state."""

    def dispatch_instruction(self, state: ScheduleState) -> Instruction | None:
        """Return the next instruction to be processed in the environment."""
        raise NotImplementedError(
            f"Backend {self.backend} has no instruction dispatch."
        )

    def add_instruction(
        self,
        instruction: Instruction,
        time: Time | None = None,
        priority: float | None = None,
    ) -> EventID:
        """Schedule a new instruction."""
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id
