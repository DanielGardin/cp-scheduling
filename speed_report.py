from pathlib import Path

from time import perf_counter
from prettytable import PrettyTable

import numpy as np

from cpscheduler import SchedulingEnv, JobShopSetup
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics import ShortestProcessingTime


benchmark_times = {
    "dmu10": 0.4,
    "dmu20": 0.8,
    "dmu30": 1.4,
    "dmu40": 2.1,
    "dmu50": 0.4,
    "dmu60": 0.8,
    "dmu70": 1.6,
    "dmu80": 2.2,
    "la10": 0.05,
    "la20": 0.05,
    "la30": 0.18,
    "la40": 0.18,
    "orb10": 0.05,
    "swv10": 0.3,
    "swv20": 1.0,
    "ta10": 0.16,
    "ta20": 0.3,
    "ta30": 0.4,
    "ta40": 0.6,
    "ta50": 0.8,
    "ta60": 1.6,
    "ta70": 2.0,
    "ta80": 7.8,
    # "kopt_ops10000_m100_1" : 3 * 60
}


RESET = "\033[0m"
def colormap(
    value: float,
    min_value: float = 0,
    max_value: float = 1,
) -> str:
    clamped = max(min(value, max_value), min_value)
    norm = (clamped - min_value) / (max_value - min_value)

    rocket_rgb = np.array([
        [  5,  19,  40],   # dark purple
        [ 94,   0,  95],   # purple
        [196,  33,  72],   # red
        [249, 123,  28],   # orange
        [252, 254, 164],   # yellow
    ], dtype=float) / 255.0

    positions = np.linspace(0, 1, len(rocket_rgb))

    # Linear interpolation between palette points
    idx = np.searchsorted(positions, norm, side='right') - 1
    idx = np.clip(idx, 0, len(rocket_rgb) - 2)
    t = (norm - positions[idx]) / (positions[idx + 1] - positions[idx])
    color = (1 - t) * rocket_rgb[idx] + t * rocket_rgb[idx + 1]
    r, g, b = (color * 255).astype(int)

    # Build ANSI escape code
    color_code = f"\033[38;2;{r};{g};{b}m"

    return color_code


def is_compiled() -> bool:
    import cpscheduler.environment.env as env

    return env.__file__.endswith(".so")


def statistics(
    data: list[float],
    unit: str = "s",
    comparison: list[float] | None = None,
) -> str:
    mean = sum(data) / len(data)
    std = (sum((t - mean) ** 2 for t in data) / len(data)) ** 0.5

    comp_string = f"{mean:.3f}"
    if std > 0:
        comp_string += f" ± {std:.3f}"

    comp_string += f"{unit}"

    if comparison is not None:
        perc = [data_point / total for data_point, total in zip(data, comparison)]

        mean_perc = sum(perc) / len(perc)
        std_perc = (sum((p - mean_perc) ** 2 for p in perc) / len(perc)) ** 0.5

        comp_string += f" ({100*mean_perc:.2f} ± {std_perc:.2%})"

    return comp_string

root = Path(__file__).parent

def test_speed(n: int = 1, full: bool = False) -> None:
    """
    Test the speed of the Shortest Processing Time heuristic on various job shop instances.
    Parameters
    ----------
    n: int
        The number of times to run the benchmark for each instance.

    full: bool
        If True, run the benchmark times for all the environment stages.
    """

    print(f"is_compiled={is_compiled()}")
    print()

    if full:
        columns = [
            "Instance",
            "All time",
            "Instance read time",
            "Setup time",
            "Reset time",
            "PDR time",
            "Step time",
            "Benchmark",
        ]
    
    else:
        columns = [
            "Instance",
            "Benchmark",
            "Step time",
        ]

    table = PrettyTable(columns)

    values: list[float] = []
    speedup_strs: list[str] = []

    for instance_name, bench_time in benchmark_times.items():
        instance_path = root / "instances/jobshop" / f"{instance_name}.txt"

        time_dict: dict[str, list[float]] = {
            "instance": [],
            "setup": [],
            "reset": [],
            "pdr": [],
            "step": [],
            "all": [],
        }

        for _ in range(n):
            global_tick = perf_counter()

            tick = perf_counter()
            instance, metadata = read_jsp_instance(instance_path)
            tock = perf_counter()
            time_dict['instance'].append(tock - tick)

            tick = perf_counter()
            env = SchedulingEnv(JobShopSetup())
            env.set_instance(instance, processing_times="processing_time")
            tock = perf_counter()
            time_dict['setup'].append(tock - tick)

            spt_agent = ShortestProcessingTime()

            tick = perf_counter()
            obs, info = env.reset()
            tock = perf_counter()
            time_dict["reset"].append(tock - tick)

            tick = perf_counter()
            action = spt_agent(obs)
            tock = perf_counter()
            time_dict["pdr"].append(tock - tick)

            tick = perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            time_dict["step"].append(perf_counter() - tick)

            global_tock = perf_counter()
            time_dict["all"].append(global_tock - global_tick)

            del env

        speedups = [bench_time / t - 1 for t in time_dict["step"]]
        mean_speedup = sum(speedups) / len(speedups)
        std_speedup = (
            sum((s - mean_speedup) ** 2 for s in speedups) / len(speedups)
        ) ** 0.5

        values.append(mean_speedup)
        speedup_strs.append(
            f"{100*mean_speedup:.2f} ± {std_speedup:.2%}"
            if std_speedup > 0
            else f"{mean_speedup:.2%}"
        )

        if full:
            datas = {
                stage: statistics(
                    time_dict[stage],
                    unit="s",
                    comparison=time_dict["all"] if stage != "all" else None,
                )
                for stage in time_dict
            }

            table.add_row(
                [
                    instance_name,
                    datas["all"],
                    datas["instance"],
                    datas["setup"],
                    datas["reset"],
                    datas["pdr"],
                    datas["step"],
                    f"{bench_time:.2f} s",
                ]
            )

        else:
            mean_time = sum(time_dict["step"]) / len(time_dict["step"])
            std_time = (sum((t - mean_time) ** 2 for t in time_dict["step"]) / len(time_dict["step"])) ** 0.5

            table.add_row(
                [
                    instance_name,
                    f"{bench_time:.2f}s",
                    (
                        f"{mean_time:.3f} ± {std_time:.3f} s"
                        if std_time > 0
                        else f"{mean_time:.3f}s"
                    ),
                ]
            )

    # RECORDS: 32x speedup for small instances
    speedup_strs = [
        colormap(value, min_value=0, max_value=8) + speed + RESET
        for value, speed in zip(values, speedup_strs)
    ]

    table.add_column("Speedup", speedup_strs)

    table._set_markdown_style()
    print(table)


if __name__ == "__main__":
    import tyro

    tyro.cli(
        test_speed,
        description="Test the speed of the Shortest Processing Time heuristic on various job shop instances.",
    )
