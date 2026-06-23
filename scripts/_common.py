from pathlib import Path
from typing import Literal, get_args

from cpscheduler import __compiled__, __version__
from cpscheduler.heuristics.pdrs import (
    MostOperationsRemaining,
    MostWorkRemaining,
    PriorityDispatchingRule,
    RandomPriority,
    ShortestProcessingTime,
)

PDR_NAMES = Literal[
    "rng",
    "spt",
    "mor",
    "mwr",
]

PDRS: dict[str, type[PriorityDispatchingRule]] = {
    "rng": RandomPriority,
    "spt": ShortestProcessingTime,
    "mor": MostOperationsRemaining,
    "mwr": MostWorkRemaining,
}

assert set(PDRS.keys()) == set(get_args(PDR_NAMES)), (
    "PDR_NAMES must match keys of pdrs dictionary"
)

SCRIPT_PATH = Path(__file__).parent
ROOT = SCRIPT_PATH.parent

OK = "\033[92m"
FAIL = "\033[91m"
WARNING = "\033[93m"
RESET = "\033[0m"

COLORMAP = [
    (5, 19, 40),
    (22, 15, 51),
    (40, 11, 62),
    (58, 7, 73),
    (76, 3, 84),
    (94, 0, 95),
    (114, 6, 90),
    (134, 13, 85),
    (155, 19, 81),
    (175, 26, 76),
    (196, 33, 72),
    (206, 51, 63),
    (217, 69, 54),
    (227, 87, 45),
    (238, 105, 36),
    (249, 123, 28),
    (249, 149, 55),
    (250, 175, 82),
    (250, 201, 109),
    (251, 227, 136),
    (252, 254, 164),
]


def _ok_fail(condition: bool) -> str:
    return f"{OK}[PASS]{RESET}" if condition else f"{FAIL}[FAIL]{RESET}"


def to_ansi_color(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def print_header() -> None:
    print(f"cpscheduler v{__version__}")
    print(f"{_ok_fail(__compiled__)} compiled")

    instance_present = (ROOT / "instances/jobshop").exists()
    print(f"{_ok_fail(instance_present)} instance directory")

    if not instance_present:
        print(
            f"\n{WARNING}Warning{RESET}: Instance directory not found. "
            "Please ensure that the 'instances/jobshop' directory exists and "
            f"contains the necessary instance files for benchmarking.\n"
        )


def mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def std(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0

    mean_value = mean(data)
    return float(
        (sum(((x - mean_value) ** 2 for x in data)) / (len(data) - 1)) ** 0.5
    )


UNITS = ("", "K", "M", "B", "T")


def format_big_number(num: list[float]) -> str:
    mean_value = mean(num)
    std_value = std(num) if len(num) > 1 else 0.0

    i = 0
    while abs(mean_value) >= 1000 and i < len(UNITS) - 1:
        mean_value /= 1000
        std_value /= 1000

        i += 1

    unit = UNITS[i]
    if std_value == 0:
        if mean_value.is_integer():
            return f"{mean_value:>3.0f}{unit}"

        return f"{mean_value:.2f}{unit}"

    if mean_value.is_integer():
        return f"{int(mean_value)} ± {int(std_value)} {unit}"

    return f"{mean_value:.2f} ± {std_value:.2f} {unit}"


def format_memory(bytes_: float) -> str:
    i = 0
    while bytes_ >= 1024 and i < len(UNITS) - 1:
        bytes_ /= 1024

        i += 1

    unit = f"{UNITS[i]}B"
    return f"{bytes_:.1f} {unit}"


TIMES = ("s ", "ms", "µs", "ns")


def format_time(seconds: list[float]) -> str:
    mean_value = mean(seconds)
    std_val = std(seconds) if len(seconds) > 1 else 0.0

    i = 0
    while mean_value < 1 and i < len(TIMES) - 1:
        mean_value *= 1000
        std_val *= 1000

        i += 1

    unit = TIMES[i]
    if std_val > 0:
        return f"{mean_value:.2f} ± {std_val:.2f} {unit}"

    return f"{mean_value:6.2f} {unit}"


def format_percentage(value: list[float]) -> str:
    mean_value = mean(value)
    std_val = std(value)

    if std_val > 0:
        return f"({100 * mean_value:.2f} ± {100 * std_val:.2f})%"

    return f"{100 * mean_value:.2f}%"
