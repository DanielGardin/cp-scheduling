from typing import Any
from numpy.typing import NDArray

import plotly.graph_objects as go
import colorcet as cc

from io import BytesIO
from PIL import Image
from numpy import array, floating

from abc import ABC, abstractmethod

from .tasks import Tasks

class Renderer(ABC):
    def __init__(
            self,
            tasks: Tasks,
            n_machines: int
        ):
        self.tasks = tasks
        self.n_machines = n_machines

    @abstractmethod
    def build_gantt(self, current_time: int) -> Any:
        ...

    @abstractmethod
    def plot(self, current_time: int) -> None:
        ...
    
    @abstractmethod
    def image(self, current_time: int) -> NDArray[floating[Any]]:
        ...


class PlotlyRenderer(Renderer):
    def build_gantt(self, current_time: int) -> go.Figure:
        fig = go.Figure()

        start_times: list[int] = []
        durations: list[int]   = []
        machines: list[int]    = []
        parts: list[int]       = []
        task_ids: list[int]    = []
        palette = cc.glasbey_dark[:len(self.tasks.jobs)]
        template = "Task %{customdata[0]} [%{customdata[1]}]:<br>Start (duration): %{customdata[2]} (%{customdata[3]})<br>Machine: %{y}<extra></extra>"

        for job, tasks in self.tasks.jobs.items():
            start_times.clear()
            durations.clear()
            machines.clear()
            parts.clear

            for task in tasks:
                for part in range(task.n_parts):
                    if not task.is_fixed(part):
                        break

                    start_times.append(task.get_start(part))
                    durations.append(task.durations[part])
                    machines.append(task.assignments[part])
                    task_ids.append(task.task_id)
                    parts.append(part)

            fig.add_trace(go.Bar(
                x=durations,
                y=machines,
                base=start_times,
                orientation='h',
                name=f"Job {job}",
                customdata=[(
                    task_ids[i],
                    parts[i],
                    start_times[i],
                    durations[i]
                ) for i in range(len(start_times))],
                hovertemplate=template,
                marker=dict(color=palette[job], line=dict(color='white', width=0.5))
            ))

        fig.update_layout(
            width=1600,
            height=800,
            barmode='overlay',
            yaxis=dict(title='Assignment', tickvals=list(range(self.n_machines)), autorange='reversed'),
            xaxis=dict(title='Time', range=[0, max(current_time / 0.95, 1)], showgrid=True, gridcolor='rgba(0,0,0,0.4)')
        )

        if len(self.tasks.jobs) <= 30:
            fig.update_layout(legend_title_text="Task jobs")

        return fig

    def plot(self, current_time: int) -> None:
        fig = self.build_gantt(current_time)
        fig.show()
    
    def image(self, current_time: int) -> NDArray[floating[Any]]:
        fig = self.build_gantt(current_time)
        img_bytes = fig.to_image(format="png")

        return array(Image.open(BytesIO(img_bytes)))