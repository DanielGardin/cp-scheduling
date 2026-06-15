"""Renderer for visualizing task schedules using the Plotly backend."""

from typing import TYPE_CHECKING, Any

from cpscheduler.environment.render import GLASBEY_BW_PALETTE, Renderer
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

        start_times: list[int] = []
        durations: list[int] = []
        machines: list[int] = []
        task_ids: list[int] = []
        template = (
            "Task %{customdata[0]} [Job %{customdata[1]}]:<br>"
            "Period: %{customdata[2]}-%{customdata[3]}<br>"
            "Machine: %{y}<extra></extra>"
        )
        instance = state.instance
        history = state.runtime.history

        for job_id, job_tasks in enumerate(instance.job_tasks):
            start_times.clear()
            durations.clear()
            machines.clear()
            task_ids.clear()

            for task_id in job_tasks:
                for entry in history[task_id]:
                    start_times.append(entry.start_time)
                    durations.append(entry.end_time - entry.start_time)
                    machines.append(entry.machine_id)
                    task_ids.append(task_id)

            customdata: Any = [
                [
                    task_ids[i],
                    job_id,
                    start_times[i],
                    start_times[i] + durations[i],
                ]
                for i in range(len(start_times))
            ]

            fig.add_trace(
                Bar(
                    x=durations,
                    y=machines,
                    base=start_times,
                    orientation="h",
                    name=f"Job {job_id}",
                    customdata=customdata,
                    hovertemplate=template,
                    marker={
                        "color": GLASBEY_BW_PALETTE[job_id % 256],
                        "line": {"color": "white", "width": 0.5},
                    },
                )
            )

        max_time = max(float(state.time) / 0.95, 1.0)

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
