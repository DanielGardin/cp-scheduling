"""Renderer for visualizing task schedules using the Plotly backend."""

from typing import TYPE_CHECKING

from cpscheduler.environment.render import Renderer
from cpscheduler.environment.render.utils import (
    GLASBEY_BW_PALETTE,
    iter_task_intervals,
)
from cpscheduler.environment.state import ScheduleState

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


class PlotlyRenderer(Renderer):
    """Renderer for visualizing task schedules using the Plotly backend."""

    render_name = "plotly"

    def build_gantt(self, state: ScheduleState) -> "Figure":
        """Build a Gantt chart representing the schedule using Plotly."""
        try:
            from plotly.graph_objects import Bar, Figure

        except ImportError:
            raise RuntimeError(
                "PlotlyRenderer requires plotly to be installed."
            ) from None

        fig = Figure()

        template = (
            "Task %{customdata[0]} [Job %{customdata[1]}]:<br>"
            "Period: %{customdata[2]}-%{customdata[3]}<br>"
            "Machine: %{y}<extra></extra>"
        )

        for task, job, machine, start, end in iter_task_intervals(state):
            fig.add_trace(
                Bar(
                    x=[end - start],
                    y=[machine],
                    base=[start],
                    orientation="h",
                    customdata=(int(task), int(job), int(start), int(end)),
                    hovertemplate=template,
                    marker={
                        "color": GLASBEY_BW_PALETTE[job % 256],
                        "line": {"color": "white", "width": 0.5},
                    },
                )
            )

        max_time = max(int(state.get_latest_end()), 1)

        fig.update_layout(
            width=1600,
            height=800,
            barmode="overlay",
            yaxis={
                "title": "Assignment",
                "tickvals": list(range(state.n_machines)),
                "autorange": "reversed",
            },
            xaxis={
                "title": "Time",
                "range": (0, max_time),
                "showgrid": True,
                "gridcolor": "rgba(0,0,0,0.4)",
            },
        )

        if state.n_jobs <= 30:
            fig.update_layout(legend_title_text="Task jobs")

        return fig

    def render(self, state: ScheduleState) -> None:
        """Render the schedule state using Plotly."""
        fig = self.build_gantt(state)
        fig.show()
