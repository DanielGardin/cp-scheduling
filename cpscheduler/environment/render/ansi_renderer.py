"""ANSI Renderer for basic visualization."""

from typing_extensions import override

from cpscheduler.environment.render.base import GLASBEY_BW_PALETTE, Renderer
from cpscheduler.environment.state import ScheduleState

LABEL_WIDTH = 6
PADDING = " " * LABEL_WIDTH
JOB_START_CHAR = "▐"
BLOCK = "█"

ANSI_RESET = "\033[0m"

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a color hex code into a RGB tuple."""
    hex_color = hex_color.removeprefix("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )

def ansi_fg(hex_color: str) -> str:
    """Convert a color hex cde into ANSI foreground color code."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m"


def ansi_bg(hex_color: str) -> str:
    """Convert a color hex cde into ANSI background color code."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[48;2;{r};{g};{b}m"

_SUBSCRIPT_DIGITS = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")

def to_subscript(n: int) -> str:
    """Convert an integer to its Unicode subscript representation."""
    return str(n).translate(_SUBSCRIPT_DIGITS)

class AnsiRenderer(Renderer):
    r"""Renderer for visualizing task schedules using ANSI character set.

    This render works better in small instances.
    A good heuristic is to select a width such that:

    width > \sum_j p_j / (m * \min_j p_j),

    where p_j is the set of processing times and m is the number of machines.
    """

    render_name = "ansi"

    width: int
    max_n_ticks: int
    color: bool

    # Maximum percentage of the current time, e.g.
    # ▐████████▐█████  ▐█
    # ├────────────────────▶
    # 0                  ^ (<=95% of the gantt width)
    cursor_max_mult = 0.95

    def __init__(
        self,
        width: int = 80,
        max_n_ticks: int | None = None,
        color: bool = True
    ):
        """Initialize an AnsiRenderer.

        Parameters
        ----------
        width: int, optional
            The maximum width to render. Defaults to 80.

        max_n_ticks: int, optional
            The maximum number of ticks in the time axis.
            Defaults to width.

        color: bool, optional
            Whether the output will use ANSI colors, or not.
            Defaults to True.

        Note
        ----
        Despite max_n_ticks being too big, the renderer will always select the
        best number of ticks automatically to maximize readability.
        """
        self.width = width
        self.max_n_ticks = max_n_ticks if max_n_ticks is not None else width
        self.color = color


    def _build_axis(
        self,
        gantt_width: int,
        block_per_time: float,
        time_window: float,
        current_time: int
    ) -> tuple[list[str], list[str]]:
        ticks = ["─"] * gantt_width
        ticklabels = [" "] * gantt_width

        ticks[0] = "├"
        ticks[-1] = "▶"
        ticklabels[0] = "0"

        # Require a double space between ticks for readability
        max_char_tick = 2 + len(str(int(time_window)))

        # We sum one because the "last tick" is ignored in favor of ▶
        n_ticks = min(self.max_n_ticks+1, gantt_width//max_char_tick)
        n_ticks = n_ticks or 1

        blocks_per_tick = gantt_width // n_ticks
        time_per_tick = int(time_window / n_ticks)

        cursor = int(current_time * block_per_time)
        ticklabels[cursor] = "^"
        for tick in range(1, n_ticks):
            pos = tick*blocks_per_tick

            ticks[pos] = "┼"

            label = str(time_per_tick*tick)

            start_pos = pos-len(label)//2
            if any(
                ticklabels[start_pos+i] != " "
                for i in range(len(label))
            ):
                continue

            for i, digit in enumerate(label, start=start_pos):
                ticklabels[i] = digit

        return ticks, ticklabels

    @override
    def build_gantt(self, state: ScheduleState) -> str:
        current_time = int(state.time)
        makespan = int(state.runtime.last_completion_time)

        time_window = max(
            current_time/self.cursor_max_mult,
            makespan,
            self.width - LABEL_WIDTH
        )

        block_per_time = (self.width - LABEL_WIDTH) / time_window

        gantt_width = int(block_per_time * time_window)

        machine_chars = [
            [" "] * gantt_width
            for _ in range(state.n_machines)
        ]

        history = state.runtime.history

        for task_id, task_history in enumerate(history):
            if not task_history:
                continue

            if self.color:
                job_id = state.get_job(task_id)
                job_color = ansi_fg(GLASBEY_BW_PALETTE[job_id % 256])

            else:
                job_color = ""

            for entry in history[task_id]:
                start = int(int(entry.start_time) * block_per_time)
                end = min(int(int(entry.end_time) * block_per_time), gantt_width)
                machine = entry.machine_id

                machine_chars[machine][start] = f"{job_color}{JOB_START_CHAR}"

                for slot in range(start+1, end):
                    machine_chars[machine][slot] = BLOCK

                if self.color:
                    pos = end-1 if end > start else start
                    machine_chars[machine][pos] += ANSI_RESET

        ansi_reset = ANSI_RESET if self.color else ""
        ticks, ticklabels = self._build_axis(
            gantt_width=gantt_width,
            block_per_time=block_per_time,
            time_window=time_window,
            current_time=state.time
        )

        return (
            "\n".join([
                f"{f'M{to_subscript(m_id)}':<{LABEL_WIDTH}}" + "".join(chars) + ansi_reset
                for m_id, chars in enumerate(machine_chars)
            ]) +
            f"\n{PADDING}" + "".join(ticks) +
            f"\n{PADDING}" + "".join(ticklabels)
        )

    def render(self, state: ScheduleState) -> None:
        """Print the rendered schedule in the terminal."""
        print(self.build_gantt(state))
