"""Rendering helpers for scheduling environments.

Renderers are responsible for visualizing schedules, typically as Gantt charts.

Subclasses should set `render_name` to register themselves automatically.
The module also exposes a color palette `GLASBEY_BW_PALETTE` useful for
consistent job coloring across renderers.
"""

from typing import Any, ClassVar

from mypy_extensions import mypyc_attr

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.state import ScheduleState

renderers: dict[str, "Renderer"] = {}


@mypyc_attr(native_class=True, allow_interpreted_subclasses=True)
class Renderer(EzPickle):
    """Base renderer for visualizing schedules.

    Subclasses should implement `build_gantt` and `render`.
    To register a renderer for `get_renderer` lookups, set the `render_name`
    class variable on the subclass.
    """

    render_name: ClassVar[str | None] = None

    def __init_subclass__(cls) -> None:
        """Automatically register subclasses with a `render_name` for lookup in `renderers`."""
        if cls.render_name is not None:
            if cls.render_name in renderers:
                raise ValueError(
                    f"Renderer name '{cls.render_name}' is already registered."
                )

            renderers[cls.render_name] = cls()

    @classmethod
    def get_renderer(cls, render_mode: str | None) -> "Renderer":
        """Return a renderer for `render_mode`.

        Parameters
        ----------
        render_mode : str | None
            Name of a registered renderer. If `None`, returns a default
            `Renderer` instance (no-op).

        """
        if render_mode is None:
            return Renderer()

        return renderers[render_mode]

    def build_gantt(self, state: ScheduleState) -> Any:
        """Construct a figure-like object representing the schedule.

        Implementations return an object understood by `render` (for example,
        a Plotly figure). The exact type is renderer-specific.
        """

    def render(self, state: ScheduleState) -> None:
        """Render the schedule to the selected backend (side-effect).

        Typical implementations will build the Gantt via `build_gantt` and
        display or save the resulting figure.
        """
