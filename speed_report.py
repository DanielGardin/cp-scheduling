from pathlib import Path

from typing import Annotated
from collections.abc import Sequence

from time import perf_counter
from prettytable import PrettyTable

import numpy as np

import tyro
from tyro.conf import arg

from cpscheduler import SchedulingEnv, JobShopSetup, __compiled__
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics._pdr import ShortestProcessingTime
from cpscheduler.utils.array_utils import disable_numpy

root = Path(__file__).parent

OK = "\033[92m"
FAIL = "\033[91m"
WARNING = "\033[93m"


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
}

UNITS: list[str] = ["s", "ms", "µs", "ns"]


def is_instance_present() -> bool:
    return (Path(__file__).parent / "instances/jobshop").exists()


RESET = "\033[0m"


def colormap(
    value: float,
    min_value: float = 0,
    max_value: float = 1,
) -> str:
    clamped = max(min(value, max_value), min_value)
    norm = (clamped - min_value) / (max_value - min_value)

    rocket_rgb = (
        np.array(
            [
                [5, 19, 40],  # dark purple
                [94, 0, 95],  # purple
                [196, 33, 72],  # red
                [249, 123, 28],  # orange
                [252, 254, 164],  # yellow
            ],
            dtype=float,
        )
        / 255.0
    )

    positions = np.linspace(0, 1, len(rocket_rgb))

    # Linear interpolation between palette points
    idx = np.searchsorted(positions, norm, side="right") - 1
    idx = np.clip(idx, 0, len(rocket_rgb) - 2)
    t = (norm - positions[idx]) / (positions[idx + 1] - positions[idx])
    color = (1 - t) * rocket_rgb[idx] + t * rocket_rgb[idx + 1]
    r, g, b = (color * 255).astype(int)

    # Build ANSI escape code
    color_code = f"\033[38;2;{r};{g};{b}m"

    return color_code


def mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def std(data: list[float]) -> float:
    mean_value = mean(data)
    return float((sum(((x - mean_value) ** 2 for x in data)) / len(data)) ** 0.5)


def statistics(
    data: list[float],
    comparison: list[float] | None = None,
) -> str:
    mean_value = mean(data)
    std_val = std(data)

    i = 0
    while mean_value < 1 and mean_value > 0:
        mean_value *= 1000
        std_val *= 1000
        i += 1

    unit = UNITS[i]

    comp_string = f"{mean_value:>6.2f}"
    if std_val > 0:
        comp_string += f" ± {std_val:>6.2f}"

    comp_string += f" {unit}"

    if comparison is not None:
        perc = [data_point / total for data_point, total in zip(data, comparison)]

        mean_perc = sum(perc) / len(perc)
        std_perc = (sum((p - mean_perc) ** 2 for p in perc) / len(perc)) ** 0.5

        if std_perc == 0:
            comp_string += f" \033[;90m({mean_perc:.2%}"

        else:
            comp_string += f" \033[;90m({100*mean_perc:.2f} ± {std_perc:.2%}"

        comp_string += f"){RESET}"

    return comp_string


def test_speed(
    n: Annotated[int, arg(aliases=("-n",))] = 1,
    full: bool = False,
    quiet: Annotated[bool, arg(aliases=("-q",))] = False,
    plot: bool = False,
    numpy: bool = True,
) -> None:
    """
    Test the speed of the Shortest Processing Time heuristic on various job shop instances.
    Parameters
    ----------
    n: int
        The number of times to run the benchmark for each instance.

    full: bool
        If True, run the benchmark times for all the environment stages.

    quiet: bool
        If True, suppress the output of the benchmark results.

    plot: bool
        If True, plot the benchmark results.
    """

    compiled = __compiled__
    instance_present = is_instance_present()

    if not quiet:
        print(Path(__file__).parent)
        print(f"{OK + '[PASS]' if compiled else FAIL + '[FAIL]'}{RESET} compiled")
        print(
            f"{OK + '[PASS]' if instance_present else FAIL + '[FAIL]'}{RESET} instance directory"
        )

    if not instance_present:
        print()
        raise FileNotFoundError(
            "Could not locate `instances` directory. Maybe you forgot to run `git submodule update --init`?"
        )

    if full:
        columns = [
            "Instance",
            "Instance read time",
            "Setup time",
            "Reset time",
            "PDR time",
            "Step time",
            "All time",
            "Benchmark",
            "Time per task",
        ]

    else:
        columns = ["Instance", "Benchmark", "Simulation time", "Time per task"]

    table = PrettyTable(columns)

    values: list[float] = []
    speedup_strs: list[str] = []

    task_numbers: list[int] = []
    perf: list[float] = []
    perf_err: list[float] = []

    times: dict[str, dict[str, float]] = {}

    dots = 0

    if not quiet:
        print(
            f"Running \033[;36m{n}{RESET} iteration{'s' if n > 1 else ''} per instance",
            end="",
        )

    spt_agent = ShortestProcessingTime()
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

        if not quiet:
            if dots < 3:
                print(".", end="", flush=True)
                dots += 1

            else:
                print(
                    f"\r{' ' * 100}",
                    end=f"\rRunning \033[;36m{n}{RESET} iteration{'s' if n > 1 else ''} per instance",
                    flush=True,
                )
                dots = 0

        n_tasks = 0
        for i in range(n):
            global_tick = perf_counter()

            tick = perf_counter()
            instance, metadata = read_jsp_instance(instance_path)
            tock = perf_counter()
            time_dict["instance"].append(tock - tick)

            tick = perf_counter()
            env = SchedulingEnv(JobShopSetup())
            env.set_instance(instance, processing_times="processing_time")
            tock = perf_counter()
            time_dict["setup"].append(tock - tick)

            tick = perf_counter()
            obs, info = env.reset()
            tock = perf_counter()
            time_dict["reset"].append(tock - tick)

            if numpy:
                tick = perf_counter()
                action = spt_agent(obs)
                tock = perf_counter()

            else:
                with disable_numpy():
                    tick = perf_counter()
                    action = spt_agent(obs)
                    tock = perf_counter()

            tick = perf_counter()
            action = spt_agent(obs)
            tock = perf_counter()
            time_dict["pdr"].append(tock - tick)

            tick = perf_counter()
            env.step(action)
            time_dict["step"].append(perf_counter() - tick)

            global_tock = perf_counter()
            time_dict["all"].append(global_tock - global_tick)

            n_tasks = env.tasks.n_tasks

            del env

        times[instance_name] = {stage: mean(time_dict[stage]) for stage in time_dict}

        mean_time = mean(time_dict["all"])
        std_time = std(time_dict["all"])

        task_numbers.append(n_tasks)
        perf.append(mean_time)
        perf_err.append(std_time)

        speedups = [bench_time / t - 1 for t in time_dict["all"]]
        mean_speedup = mean(speedups)
        std_speedup = std(speedups)

        values.append(mean_speedup)
        speedup_strs.append(
            f"{100*mean_speedup:5.2f} ± {std_speedup:5.2%}"
            if std_speedup > 0
            else f"{mean_speedup:5.2%}"
        )

        tasks_per_second = [t / n_tasks for t in time_dict["all"]]
        mean_tps = mean(tasks_per_second)
        std_tps = std(tasks_per_second)

        i = 0
        while mean_tps < 1:
            mean_tps *= 1000
            std_tps *= 1000
            i += 1

        tasks_per_second_str = (
            f"{mean_tps:5.2f} ± {std_tps:5.2f} {UNITS[i]}"
            if std_tps > 0
            else f"{mean_tps:5.2f} {UNITS[i]}"
        )

        if full:
            datas = {
                stage: statistics(
                    time_dict[stage],
                    comparison=time_dict["all"] if stage != "all" else None,
                )
                for stage in time_dict
            }

            table.add_row(
                [
                    instance_name,
                    datas["instance"],
                    datas["setup"],
                    datas["reset"],
                    datas["pdr"],
                    datas["step"],
                    datas["all"],
                    f"{bench_time:.2f} s",
                    tasks_per_second_str,
                ]
            )

        else:
            mean_time = mean(time_dict["all"])
            std_time = std(time_dict["all"])

            i = 0
            while mean_time < 1 and mean_time > 0:
                mean_time *= 1000
                std_time *= 1000
                i += 1

            table.add_row(
                [
                    instance_name,
                    f"{bench_time:.2f} s",
                    (
                        f"{mean_time:5.2f} ± {std_time:5.2f} {UNITS[i]}"
                        if std_time > 0
                        else f"{mean_time:5.2f} {UNITS[i]}"
                    ),
                    tasks_per_second_str,
                ]
            )

    # RECORDS: 49x speedup for small instances
    #          10x speedup for large instances
    speedup_strs = [
        colormap(value, min_value=0, max_value=12) + speed + RESET
        for value, speed in zip(values, speedup_strs)
    ]

    table.add_column("Speedup", speedup_strs)

    table._set_markdown_style()

    print("\n")
    print(table, flush=True)

    if plot:
        from matplotlib.axes import Axes

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)

        ax: Sequence[Axes]
        fig, ax = plt.subplots(ncols=2, figsize=(20, 6))

        stages = ["instance", "setup", "reset", "pdr", "step"]
        stage_times = np.array(
            [[time_dict[stage] for stage in time_dict] for time_dict in times.values()]
        ).T

        for i, stage in enumerate(stages):
            ax[0].bar(
                list(times.keys()),
                stage_times[i],
                label=stage,
                linewidth=0,
                bottom=np.sum(stage_times[:i], axis=0) if i > 0 else None,
            )

        ax[0].set_title("Time per stage for each instance")
        ax[0].set_xlabel("Instance")
        ax[0].set_ylabel("Time (s)")
        ax[0].legend(title="Stage", loc="upper left")
        ax[0].set_xticks(list(times.keys()))
        ax[0].set_xticklabels(list(times.keys()), rotation=90)

        ax[1].set_title("Average time vs Number of Tasks")
        ax[1].set_xlabel("Number of Tasks")
        ax[1].set_ylabel("Average time (s)")

        # Fit a quadratic curve to the data
        fit_coef, ang_coef, lin_coef = np.polyfit(task_numbers, perf, 2)

        x = np.linspace(0, max(task_numbers), 100)
        ax[1].plot(
            x,
            lin_coef + x * (ang_coef + x * fit_coef),
            color="red",
            label="Quadratic Fit",
            zorder=1,
        )

        sup_coef = max((perf) / n_tasks**2 for perf, n_tasks in zip(perf, task_numbers))
        inf_coef = min((perf) / n_tasks**2 for perf, n_tasks in zip(perf, task_numbers))

        # Plot the area between the curves
        ax[1].fill_between(
            x,
            lin_coef + x * (ang_coef + x * inf_coef),
            lin_coef + x * (ang_coef + x * sup_coef),
            color="lightblue",
            alpha=0.5,
            label="Quadratic Growth Area",
            zorder=0,
        )

        # Scatter plot with error bars
        ax[1].errorbar(
            task_numbers,
            perf,
            yerr=perf_err,
            color="black",
            fmt="o",
            ecolor="black",
            elinewidth=2,
            capsize=4,
            label="Measured Times",
            zorder=2,
        )

        ax[1].legend()
        plt.tight_layout()

        plt.suptitle("Speed Benchmark Report", fontsize=16, fontweight="bold")
        plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the title

        plt.savefig(root / "report.pdf", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    tyro.cli(
        test_speed,
        description="Test the speed of the Shortest Processing Time heuristic on various job shop instances.",
    )
