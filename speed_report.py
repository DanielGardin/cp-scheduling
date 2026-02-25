from pathlib import Path

from typing import Annotated
from collections.abc import Sequence


from time import perf_counter
from prettytable import PrettyTable, TableStyle

import numpy as np

import tyro
from tyro.conf import arg

from cpscheduler import SchedulingEnv, JobShopSetup, __compiled__, __version__
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics._pdr import ShortestProcessingTime
from cpscheduler.utils.array_utils import disable_numpy

root = Path(__file__).parent

OK = "\033[92m"
FAIL = "\033[91m"
WARNING = "\033[93m"

def ok_fail(condition: bool) -> str:
    return (OK + "[PASS]" if condition else FAIL + "[FAIL]") + RESET


benchmark_times = {
    # "yn01": float("inf"),
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
    # "lta_j100_m100_10": 90.0,
}

RESET = "\033[0m"


COLORMAP = [
    [5, 19, 40],
    [22, 15, 51],
    [40, 11, 62],
    [58, 7, 73],
    [76, 3, 84],
    [94, 0, 95],
    [114, 6, 90],
    [134, 13, 85],
    [155, 19, 81],
    [175, 26, 76],
    [196, 33, 72],
    [206, 51, 63],
    [217, 69, 54],
    [227, 87, 45],
    [238, 105, 36],
    [249, 123, 28],
    [249, 149, 55],
    [250, 175, 82],
    [250, 201, 109],
    [251, 227, 136],
    [252, 254, 164],
]

def to_ansi_color(rgb: Sequence[int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"

def mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def std(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0

    mean_value = mean(data)
    return float((sum(((x - mean_value) ** 2 for x in data)) / (len(data) - 1)) ** 0.5)

def format_big_number(num: list[float]) -> str:
    mean_value = mean(num)
    std_value = std(num) if len(num) > 1 else 0.0

    for unit in ["", "K", "M"]:
        if abs(mean_value) < 1000:
            break

        mean_value /= 1000
        std_value /= 1000

    else:
        unit = "B"

    if std_value == 0:
        if mean_value.is_integer():
            return f"{int(mean_value)}{unit}"
    
        return f"{mean_value:.2f}{unit}"

    if mean_value.is_integer():
        return f"{int(mean_value)} ± {int(std_value)} {unit}"

    return f"{mean_value:.2f} ± {std_value:.2f} {unit}"

def format_memory(bytes_: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes_) < 1024:
            return f"{bytes_:.1f} {unit}"

        bytes_ /= 1024

    return f"{bytes_:.1f} TB"

def format_time(seconds: list[float]) -> str:
    mean_value = mean(seconds)
    std_val = std(seconds) if len(seconds) > 1 else 0.0

    for unit in ("s", "ms", "µs"):
        if mean_value >= 1:
            break
        
        mean_value *= 1000
        std_val *= 1000

    else:
        unit = "ns"

    if std_val > 0:
        return f"{mean_value:.2f} ± {std_val:.2f} {unit}"

    return f"{mean_value:.2f} {unit}"


def format_percentage(value: list[float]) -> str:
    mean_value = mean(value)
    std_val = std(value)

    if std_val > 0:
        return f"({100*mean_value:.2f} ± {100*std_val:.2f})%"

    return f"{100*mean_value:.2f}%"

def test_memory(
    dynamic: bool,
    numpy: bool,
    quiet: bool,
) -> None:
    """Run a single iteration per instance with tracemalloc to measure peak memory usage."""
    import tracemalloc

    spt_agent = ShortestProcessingTime(available=dynamic)

    columns = ["Instance", "Tasks", "Peak Memory", "Memory/Task"]
    table = PrettyTable(columns)
    table.set_style(TableStyle.MARKDOWN)

    if not quiet:
        print(
            f"Running memory benchmark",
            end="",
        )

    dots = 0
    for instance_name in benchmark_times:
        if not quiet:
            if dots < 3:
                print(".", end="", flush=True)
                dots += 1

            else:
                print(
                    f"\r{' ' * 100}",
                    end=f"\rRunning memory benchmark",
                    flush=True,
                )
                dots = 0

        instance_path = root / "instances/jobshop" / f"{instance_name}.txt"

        instance, _ = read_jsp_instance(instance_path)
        env = SchedulingEnv(JobShopSetup())
        env.set_instance(instance)

        tracemalloc.start()

        obs, _ = env.reset()

        if dynamic:
            done = False
            while not done:
                single_action = spt_agent(obs)[0]
                obs, _, done, _, _ = env.step(single_action)
        else:
            if numpy:
                action = spt_agent(obs)
            else:
                with disable_numpy():
                    action = spt_agent(obs)
            env.step(action)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n_tasks = env.state.n_tasks
        per_task = peak_mem / n_tasks if n_tasks > 0 else 0

        table.add_row([
            instance_name,
            n_tasks,
            format_memory(peak_mem),
            format_memory(per_task),
        ])

        del env

    print("\n")
    print(table, flush=True)


class RunResult:
    MIN_SPEEDUP = 4
    "Speedup required to be considered a significant improvement. Maps to colormap[0]."

    MAX_SPEEDUP = 20
    "Speedup considered a very significant improvement. Maps to colormap[-1]."

    def __init__(self) -> None:
        self.current_instance: str | None = None
        self.current_run = 0

        self.instance_names: list[str] = []

        self.n_tasks: dict[str, int] = {}
        self.n_events: dict[str, int] = {}

        self.instance_times: dict[str, list[float]] = {}
        self.initialization_times: dict[str, list[float]] = {}
        self.reset_times: dict[str, list[float]] = {}
        self.pdr_times: dict[str, list[float]] = {}
        self.step_times: dict[str, list[float]] = {}
        self.simulation_times: dict[str, list[float]] = {}

    def start_instance(
        self,
        instance_name: str,
        n_tasks: int,
        n_events: int,
    ) -> None:
        self.current_instance = instance_name
        self.instance_names.append(instance_name)
        self.n_tasks[instance_name] = n_tasks
        self.n_events[instance_name] = n_events

        self.current_run = 0
        self.instance_times[instance_name] = []
        self.initialization_times[instance_name] = []
        self.reset_times[instance_name] = []
        self.pdr_times[instance_name] = []
        self.step_times[instance_name] = []
        self.simulation_times[instance_name] = []
    

    def log_run(
        self,
        instance_time: float,
        setup_time: float,
        reset_time: float,
        pdr_time: float,
        step_time: float,
        simulation_time: float,
        
    ) -> None:
        if self.current_instance is None:
            raise ValueError("No instance started. Call start_instance() before logging runs.")

        instance_name = self.current_instance
        
        self.instance_times[instance_name].append(instance_time)
        self.initialization_times[instance_name].append(setup_time)
        self.reset_times[instance_name].append(reset_time)
        self.pdr_times[instance_name].append(pdr_time)
        self.step_times[instance_name].append(step_time)
        self.simulation_times[instance_name].append(simulation_time)

    def print_results(self, full: bool) -> None:
        table = PrettyTable([
            "Instance",
        ] + ([
            "Instance time",
            "Initialization time",
            "Reset time",
            "PDR time",
            "Step time",
        ] if full else []) + [
            "Simulation time",
            "Tasks",
            "Time per task",
            "Events",
            "Events/sec",
            "Speedup"
        ])
        table.set_style(TableStyle.MARKDOWN)

        for instance_name in self.instance_names:
            instance_read_time = format_time(self.instance_times[instance_name])
            self.initialization_time = format_time(self.initialization_times[instance_name])
            reset_time = format_time(self.reset_times[instance_name])
            pdr_time = format_time(self.pdr_times[instance_name])
            step_time = format_time(self.step_times[instance_name])
            simulation_time = format_time(self.simulation_times[instance_name])
            tasks = format_big_number([self.n_tasks[instance_name]])
            time_per_task = format_time(
                [t / self.n_tasks[instance_name] for t in self.simulation_times[instance_name]]
            )
            events = format_big_number([self.n_events[instance_name]])
            events_per_sec = format_big_number(
                [self.n_events[instance_name] / t for t in self.simulation_times[instance_name]]
            )

            speedups = [
                benchmark_times.get(instance_name, 0.) / t for t in self.simulation_times[instance_name]
                if t > 0
            ]

            idx = int(len(COLORMAP) * (mean(speedups) - self.MIN_SPEEDUP) / (self.MAX_SPEEDUP - self.MIN_SPEEDUP))
            idx = max(0, min(len(COLORMAP) - 1, idx))

            speedup = to_ansi_color(COLORMAP[idx]) + format_percentage(speedups) + RESET

            table.add_row([
                instance_name,
            ] + ([
                instance_read_time,
                self.initialization_time,
                reset_time,
                pdr_time,
                step_time,
            ] if full else []) + [
                simulation_time,
                tasks,
                time_per_task,
                events,
                events_per_sec,
                speedup
            ])

        print("\n")
        print(table, flush=True)

    def plot_results(self) -> None:
        from matplotlib.axes import Axes

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)

        ax: Sequence[Axes]
        fig, ax = plt.subplots(ncols=2, figsize=(18, 4), width_ratios=[2, 1])

        stages = ["instance", "initialization", "reset", "pdr", "step"]

        bottom = [0.] * len(self.instance_names)
        for stage in stages:
            stage_times = [
                mean(self.__dict__[f"{stage}_times"][instance])
                for instance in self.instance_names
            ]

            ax[0].bar(
                self.instance_names,
                stage_times,
                label=stage,
                linewidth=0,
                bottom=bottom,
            )

            bottom = [b + t for b, t in zip(bottom, stage_times)]

        ax[0].set_title("Time per stage for each instance")
        ax[0].set_xlabel("Instance")
        ax[0].set_ylabel("Time (s)")
        ax[0].legend(title="Stage", loc="upper left")
        ax[0].set_xticks(self.instance_names)
        ax[0].set_xticklabels(self.instance_names, rotation=90)

        ax[1].set_title("Average time vs Number of Tasks")
        ax[1].set_xlabel("Number of Tasks")
        ax[1].set_ylabel("Average time (s)")

        # Fit a quadratic curve to the data
        task_numbers = [self.n_tasks[instance] for instance in self.instance_names]
        perf = [mean(self.simulation_times[instance]) for instance in self.instance_names]
        perf_err = [std(self.simulation_times[instance]) for instance in self.instance_names]

        fit_coef, *_ = np.polyfit(task_numbers, perf, 2)

        x = np.linspace(0.9* min(task_numbers), 1.1 * max(task_numbers), 100)
        ax[1].plot(
            x,
            fit_coef * x**2,
            color="red",
            label="Quadratic Fit",
            zorder=1,
        )

        sup_coef = max((perf) / n_tasks**2 for perf, n_tasks in zip(perf, task_numbers))
        inf_coef = min((perf) / n_tasks**2 for perf, n_tasks in zip(perf, task_numbers))

        # Plot the area between the curves
        ax[1].fill_between(
            x,
            inf_coef * x**2,
            sup_coef * x**2,
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

        ax[1].loglog()
        ax[1].legend()

        fig.suptitle("Speed Benchmark Report", fontsize=16, fontweight="bold")
        fig.subplots_adjust(top=0.9)  # Adjust the top to make space for the title

        fig.savefig(root / "report.pdf", dpi=300)
        plt.show()


def print_header() -> None:
    print(f"cpscheduler v{__version__}")
    print(f"{ok_fail(__compiled__)} compiled")
    
    instance_present = (Path(__file__).parent / "instances/jobshop").exists()
    print(f"{ok_fail(instance_present)} instance directory")

    if not instance_present:
        print()
        raise FileNotFoundError(
            "Could not locate `instances` directory. Perhaps you forgot to run "
            "`git submodule update --init`?"
        )

def run_cli(
    n_runs: Annotated[int, arg(aliases=("-n",))] = 1,
    full: bool = False,
    quiet: Annotated[bool, arg(aliases=("-q",))] = False,
    plot: bool = False,
    numpy: bool = True,
    dynamic: bool = False,
    memory: bool = False,
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
    if not quiet:
        print_header()

    if memory:
        print("Running bechmark: memory usage")

        test_memory(dynamic=dynamic, numpy=numpy, quiet=quiet)
        return


    print("Running bechmark: time")
    result = RunResult()

    # TODO: Make the PDR a parameter
    agent = ShortestProcessingTime(available=dynamic)

    dots = 0
    if not quiet:
        print(
            f"Running \033[;36m{n_runs}{RESET} iteration{'s' if n_runs > 1 else ''} per instance",
            end="",
        )

    for instance_name, _ in benchmark_times.items():
        if not quiet:
            if dots < 3:
                print(".", end="", flush=True)
                dots += 1

            else:
                print(
                    f"\r{' ' * 100}",
                    end=f"\rRunning \033[;36m{n_runs}{RESET} iteration{'s' if n_runs > 1 else ''} per instance",
                    flush=True,
                )
                dots = 0

        instance_path = root / "instances/jobshop" / f"{instance_name}.txt"

        env = SchedulingEnv(JobShopSetup(), instance=read_jsp_instance(instance_path)[0])
        obs, _ = env.reset()
        env.step(agent(obs))

        result.start_instance(
            instance_name,
            n_tasks=env.state.n_tasks,
            n_events=env.event_count,
        )

        del env

        for _ in range(n_runs):
            global_tick = perf_counter()

            tick = perf_counter()
            instance, _ = read_jsp_instance(instance_path)
            instance_read_time = perf_counter() - tick

            tick = perf_counter()
            env = SchedulingEnv(JobShopSetup(), instance=instance)
            initialization_times = perf_counter() - tick

            tick = perf_counter()
            obs, _ = env.reset()
            reset_time = perf_counter() - tick

            if not dynamic:
                with disable_numpy(disable=not numpy):
                    tick = perf_counter()
                    action = agent(obs)
                    pdr_time = perf_counter() - tick
            
                tick = perf_counter()
                env.step(action)
                step_time = perf_counter() - tick

            else:
                pdr_time = 0.
                step_time = 0.

                done = False
                while not done:
                    tick = perf_counter()
                    single_action = agent(obs)[0]
                    pdr_time += perf_counter() - tick

                    tick = perf_counter()
                    obs, _, done, *_ = env.step(single_action)
                    step_time += perf_counter() - tick

            simulation_time = perf_counter() - global_tick

            result.log_run(
                instance_read_time,
                initialization_times,
                reset_time,
                pdr_time,
                step_time,
                simulation_time,
            )

            del env

    result.print_results(full=full)

    if plot:
        result.plot_results()


if __name__ == "__main__":
    tyro.cli(
        run_cli,
        description="Test the speed of the Shortest Processing Time heuristic on various job shop instances.",
    )
