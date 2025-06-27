"""
    render.py

    This module defines the Renderer class and its PlotlyRenderer subclass for rendering Gantt
    charts of task schedules. The Renderer class is an abstract base class that provides a common
    interface for rendering tasks.
"""
from typing import Any

from abc import ABC, abstractmethod

from mypy_extensions import mypyc_attr

from .tasks import Tasks

try:
    from plotly import graph_objects as go # type: ignore[import]
    from colorcet import glasbey_dark      # type: ignore[import]

except ImportError:
    go = None
    glasbey_dark = None

@mypyc_attr(allow_interpreted_subclasses=True)
class Renderer(ABC):
    "Renderer base class for visualizing task schedules."
    def __init__(
            self,
            tasks: Tasks
        ):
        self.tasks = tasks

    @abstractmethod
    def build_gantt(self, current_time: int) -> Any:
        "Build a figure-like object representing the Gantt chart."

    @abstractmethod
    def render(self, current_time: int) -> None:
        "Render the built Gantt chart."


class PlotlyRenderer(Renderer):
    "Renderer for visualizing task schedules using the Plotly backend."

    def build_gantt(self, current_time: int) -> Any:
        if go is None or glasbey_dark is None:
            raise ImportError(
                "Plotly and Colorcet are required for rendering Gantt charts with PlotlyRenderer. "
                "Please install them using 'pip install plotly colorcet'."
            )

        assert go is not None

        fig = go.Figure()

        start_times: list[int] = []
        durations: list[int]   = []
        machines: list[int]    = []
        parts: list[int]       = []
        task_ids: list[int]    = []
        palette = glasbey_dark[:len(self.tasks.jobs)] #type: ignore[no-redef]
        template = "Task %{customdata[0]} [Job %{customdata[1]}]:<br>"\
                   "Period: %{customdata[2]}-%{customdata[3]}<br>"\
                   "Machine: %{y}<extra></extra>"

        for job, tasks in enumerate(self.tasks.jobs):
            start_times.clear()
            durations.clear()
            machines.clear()
            parts.clear()

            for task in tasks:
                for part in range(task.n_parts):
                    if not task.is_fixed():
                        break

                    start_times.append(task.get_start(part))
                    durations.append(task.durations[part])
                    machines.append(task.assignments[part])
                    task_ids.append(task.task_id)
                    parts.append(part)

            fig.add_trace(go.Bar( # type: ignore[call-arg]
                x=durations,
                y=machines,
                base=start_times,
                orientation='h',
                name=f"Job {job}",
                customdata=[(
                    task_ids[i],
                    parts[i],
                    start_times[i],
                    start_times[i] + durations[i]
                ) for i in range(len(start_times))],
                hovertemplate=template,
                marker=dict(color=palette[job], line=dict(color='white', width=0.5)) # type: ignore[arg-type]
            ))

        fig.update_layout( # type: ignore[call-arg]
            width=1600,
            height=800,
            barmode='overlay',
            yaxis=dict(title='Assignment', tickvals=list(range(self.tasks.n_machines)), autorange='reversed'),
            xaxis=dict(title='Time', range=[0, max(current_time / 0.95, 1)], showgrid=True, gridcolor='rgba(0,0,0,0.4)')
        )

        if len(self.tasks.jobs) <= 30:
            fig.update_layout(legend_title_text="Task jobs") # type: ignore[call-arg]

        return fig

    def render(self, current_time: int) -> None:
        fig = self.build_gantt(current_time)
        fig.show()

    # This method is
    # def image(self, current_time: int) -> NDArray[floating[Any]]:
    #     if 'PIL' not in modules:
    #         raise ImportError(
    #             "PIL is required for rendering Gantt charts with PlotlyRenderer. "
    #             "Please install them using 'pip install pillow'."
    #         )

    #     fig = self.build_gantt(current_time)
    #     img_bytes = fig.to_image(format="png")

    #     return array(Image.open(BytesIO(img_bytes)))
