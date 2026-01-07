"""
render.py

This module defines the Renderer class and its PlotlyRenderer subclass for rendering Gantt
charts of task schedules. The Renderer class is an abstract base class that provides a common
interface for rendering tasks.
"""

from typing import Any, ClassVar

from cpscheduler.environment.state import ScheduleState

renderers: dict[str, "Renderer"] = {}


class Renderer:
    "Renderer base class for visualizing task schedules."

    render_name: ClassVar[str | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls.render_name is not None:
            renderers[cls.render_name] = cls()

    @classmethod
    def get_renderer(cls, render_mode: str | None) -> "Renderer":
        "Get the registered renderers."
        if render_mode is None:
            return Renderer()

        return renderers[render_mode]

    def build_gantt(self, current_time: int, state: ScheduleState) -> Any:
        "Build a figure-like object representing the Gantt chart."

    def render(self, current_time: int, state: ScheduleState) -> None:
        "Render the built Gantt chart."


try:
    from plotly import graph_objects as go
    from colorcet import glasbey_dark

    class PlotlyRenderer(Renderer):
        "Renderer for visualizing task schedules using the Plotly backend."

        name = "plotly"

        def build_gantt(self, current_time: int, state: ScheduleState) -> Any:
            if go is None or glasbey_dark is None:
                raise ImportError(
                    "Plotly and Colorcet are required for rendering Gantt charts with PlotlyRenderer. "
                    "Please install them using 'pip install plotly colorcet'."
                )

            assert go is not None

            fig = go.Figure()

            start_times: list[int] = []
            durations: list[int] = []
            machines: list[int] = []
            task_ids: list[int] = []
            palette = glasbey_dark[: state.n_jobs]
            template = (
                "Task %{customdata[0]} [Job %{customdata[1]}]:<br>"
                "Period: %{customdata[2]}-%{customdata[3]}<br>"
                "Machine: %{y}<extra></extra>"
            )

            for job, job_tasks in enumerate(state.jobs):
                for task in job_tasks:
                    for part in range(task.n_parts):
                        if not task.is_fixed():
                            break

                        start_times.append(task.get_start(part))
                        durations.append(task.durations[part])
                        machines.append(task.assignments[part])
                        task_ids.append(task.task_id)

                fig.add_trace(
                    go.Bar(
                        x=durations,
                        y=machines,
                        base=start_times,
                        orientation="h",
                        name=f"Job {job}",
                        customdata=[
                            (
                                task_ids[i],
                                job,
                                start_times[i],
                                start_times[i] + durations[i],
                            )
                            for i in range(len(start_times))
                        ],
                        hovertemplate=template,
                        marker=dict(
                            color=palette[job], line=dict(color="white", width=0.5)
                        ),
                    )
                )

            max_time = int(current_time / 0.95)
            if max_time < 1:
                max_time = 1

            fig.update_layout(
                width=1600,
                height=800,
                barmode="overlay",
                yaxis=dict(
                    title="Assignment",
                    tickvals=list(range(state.n_machines)),
                    autorange="reversed",
                ),
                xaxis=dict(
                    title="Time",
                    range=(0, max_time),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.4)",
                ),
            )

            if state.n_jobs <= 30:
                fig.update_layout(legend_title_text="Task jobs")

            return fig

        def render(self, current_time: int, state: ScheduleState) -> None:
            fig = self.build_gantt(current_time, state)
            fig.show()

except ImportError:
    PLOTLY_AVAILABLE = False

else:
    PLOTLY_AVAILABLE = True
