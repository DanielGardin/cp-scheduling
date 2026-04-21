"""
render.py

This module defines the Renderer class and its PlotlyRenderer subclass for rendering Gantt
charts of task schedules. The Renderer class is an abstract base class that provides a common
interface for rendering tasks.
"""

from typing import Any, ClassVar

from cpscheduler.environment.state import ScheduleState
from cpscheduler.environment.constants import EzPickle

renderers: dict[str, "Renderer"] = {}


class Renderer(EzPickle):
    "Renderer base class for visualizing task schedules."

    render_name: ClassVar[str | None] = None

    def __init_subclass__(cls) -> None:
        if cls.render_name is not None:
            if cls.render_name in renderers:
                raise ValueError(
                    f"Renderer name '{cls.render_name}' is already registered."
                )

            renderers[cls.render_name] = cls()

    @classmethod
    def get_renderer(cls, render_mode: str | None) -> "Renderer":
        "Get the registered renderers."
        if render_mode is None:
            return Renderer()

        return renderers[render_mode]

    def build_gantt(self, state: ScheduleState) -> Any:
        "Build a figure-like object representing the Gantt chart."

    def render(self, state: ScheduleState) -> None:
        "Render the built Gantt chart."


GLASBEY_BW_PALETTE: list[str] = [
    "#d60000",
    "#8c3bff",
    "#018700",
    "#00acc6",
    "#e6a500",
    "#ff7ed1",
    "#6b004f",
    "#573b00",
    "#005659",
    "#15e18c",
    "#0000dd",
    "#a17569",
    "#bcb6ff",
    "#bf03b8",
    "#645472",
    "#790000",
    "#0774d8",
    "#729a7c",
    "#ff7752",
    "#004b00",
    "#8e7b01",
    "#f2007b",
    "#8eba00",
    "#a57bb8",
    "#5901a3",
    "#e2afaf",
    "#a03a52",
    "#a1c8c8",
    "#9e4b00",
    "#546744",
    "#bac389",
    "#5e7b87",
    "#60383b",
    "#8287ff",
    "#380000",
    "#e252ff",
    "#2f5282",
    "#7ecaff",
    "#c4668e",
    "#008069",
    "#919eb6",
    "#cc7407",
    "#7e2a8e",
    "#00bda3",
    "#2db152",
    "#4d33ff",
    "#00e400",
    "#ff00cd",
    "#c85748",
    "#e49cff",
    "#1ca1ff",
    "#6e70aa",
    "#c89a69",
    "#77563b",
    "#03dae6",
    "#c1a3c3",
    "#ff6989",
    "#ba00fd",
    "#915280",
    "#9e0174",
    "#93a14f",
    "#364424",
    "#af6dff",
    "#596d00",
    "#ff3146",
    "#828056",
    "#006d2d",
    "#8956af",
    "#5949a3",
    "#773416",
    "#85c39a",
    "#5e1123",
    "#d48580",
    "#a32818",
    "#0087b1",
    "#ca0044",
    "#ffa056",
    "#eb4d00",
    "#6b9700",
    "#528549",
    "#755900",
    "#c8c33f",
    "#91d370",
    "#4b9793",
    "#4d230c",
    "#60345b",
    "#8300cf",
    "#8a0031",
    "#9e6e31",
    "#ac8399",
    "#c63189",
    "#015438",
    "#086b83",
    "#87a8eb",
    "#6466ef",
    "#c35dba",
    "#019e70",
    "#805059",
    "#826e8c",
    "#b3bfda",
    "#b89028",
    "#ff97b1",
    "#a793e1",
    "#698cbd",
    "#4b4f01",
    "#4801cc",
    "#60006e",
    "#446966",
    "#9c5642",
    "#7bacb5",
    "#cd83bc",
    "#0054c1",
    "#7b2f4f",
    "#fb7c00",
    "#34bf00",
    "#ff9c87",
    "#e1b669",
    "#526077",
    "#5b3a7c",
    "#eda5da",
    "#ef52a3",
    "#5d7e69",
    "#c3774f",
    "#d14867",
    "#6e00eb",
    "#1f3400",
    "#c14103",
    "#6dd4c1",
    "#46709e",
    "#a101c3",
    "#0a8289",
    "#afa501",
    "#a55b6b",
    "#fd77ff",
    "#8a85ae",
    "#c67ee8",
    "#9aaa85",
    "#876bd8",
    "#01baf6",
    "#af5dd1",
    "#59502a",
    "#b5005e",
    "#7cb569",
    "#4985ff",
    "#00c182",
    "#d195aa",
    "#a34ba8",
    "#e205e2",
    "#16a300",
    "#382d00",
    "#832f33",
    "#5d95aa",
    "#590f00",
    "#7b4600",
    "#6e6e31",
    "#335726",
    "#4d60b5",
    "#a19564",
    "#623f28",
    "#44d457",
    "#70aacf",
    "#2d6b4d",
    "#72af9e",
    "#fd1500",
    "#d8b391",
    "#79893b",
    "#7cc6d8",
    "#db9036",
    "#eb605d",
    "#eb5ed4",
    "#e47ba7",
    "#a56b97",
    "#009744",
    "#ba5e21",
    "#bcac52",
    "#87d82f",
    "#873472",
    "#aea8d1",
    "#e28c62",
    "#d1b1eb",
    "#36429e",
    "#3abdc1",
    "#669c4d",
    "#9e0399",
    "#4d4d79",
    "#7b4b85",
    "#c33431",
    "#8c6677",
    "#aa002d",
    "#7e0175",
    "#01824d",
    "#724967",
    "#727790",
    "#6e0099",
    "#a0ba52",
    "#e16e31",
    "#c46970",
    "#6d5b95",
    "#a33b74",
    "#316200",
    "#87004f",
    "#335769",
    "#ba8c7c",
    "#1859ff",
    "#909101",
    "#2b8ad4",
    "#1626ff",
    "#21d3ff",
    "#a390af",
    "#8a6d4f",
    "#5d213d",
    "#db03b3",
    "#6e56ca",
    "#642821",
    "#ac7700",
    "#a3bff6",
    "#b58346",
    "#9738db",
    "#b15093",
    "#7242a3",
    "#878ed1",
    "#8970b1",
    "#6baf36",
    "#5979c8",
    "#c69eff",
    "#56831a",
    "#00d6a7",
    "#824638",
    "#11421c",
    "#59aa75",
    "#905b01",
    "#f64470",
    "#ff9703",
    "#e14231",
    "#ba91cf",
    "#34574d",
    "#f7807c",
    "#903400",
    "#b3cd00",
    "#2d9ed3",
    "#798a9e",
    "#50807c",
    "#c136d6",
    "#eb0552",
    "#b8ac7e",
    "#487031",
    "#839564",
    "#d89c89",
    "#0064a3",
    "#4b9077",
    "#8e6097",
    "#ff5238",
    "#a7423b",
    "#006e70",
    "#97833d",
    "#dbafc8",
]

class PlotlyRenderer(Renderer):
    "Renderer for visualizing task schedules using the Plotly backend."

    render_name = "plotly"

    def build_gantt(self, state: ScheduleState) -> Any:
        from plotly import graph_objects as go

        fig = go.Figure()

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
            for task_id in job_tasks:
                for entry in history[task_id]:

                    start_times.append(entry.start_time)
                    durations.append(entry.end_time - entry.start_time)
                    machines.append(entry.machine_id)
                    task_ids.append(task_id)

            fig.add_trace(
                go.Bar(
                    x=durations,
                    y=machines,
                    base=start_times,
                    orientation="h",
                    name=f"Job {job_id}",
                    customdata=[
                        (
                            task_ids[i],
                            job_id,
                            start_times[i],
                            start_times[i] + durations[i],
                        )
                        for i in range(len(start_times))
                    ],
                    hovertemplate=template,
                    marker=dict(
                        color=GLASBEY_BW_PALETTE[job_id % 256],
                        line=dict(color="white", width=0.5),
                    ),
                )
            )

        max_time = max(float(state.time) / 0.95, 1.0)

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

    def render(self, state: ScheduleState) -> None:
        fig = self.build_gantt(state)
        fig.show()
