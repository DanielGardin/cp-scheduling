from pathlib import Path

from typing import Annotated, Literal, get_args
from collections.abc import Sequence

import gc

from math import exp

from time import perf_counter
from prettytable import PrettyTable, TableStyle

import tyro
from tyro.conf import arg

from cpscheduler import SchedulingEnv, JobShopSetup, Makespan, __compiled__, __version__
from cpscheduler.gym import SchedulingEnvGym

from cpscheduler.instances import read_jsp_instance

from cpscheduler.heuristics.pdrs import (
    PriorityDispatchingRule,
    RandomPriority,
    ShortestProcessingTime,
    MostOperationsRemaining,
    MostWorkRemaining,
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

assert set(PDRS.keys()) == set(get_args(PDR_NAMES)), "PDR_NAMES must match keys of pdrs dictionary"

SCRIPT_PATH = Path(__file__).parent
ROOT = SCRIPT_PATH.parent

OK = "\033[92m"
FAIL = "\033[91m"
WARNING = "\033[93m"
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

def ok_fail(condition: bool) -> str:
    return (OK + "[PASS]" if condition else FAIL + "[FAIL]") + RESET

def get_gray_ansi(percentage: float) -> str:
    gray_value = int(128 + 127 * percentage)
    return f"\033[38;2;{gray_value};{gray_value};{gray_value}m"

def to_ansi_color(rgb: Sequence[int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"

# These times are based on the results of the CPEnv implementation by Tassel, Pierre
# Environment compiled: https://github.com/ingambe/JobShopCPEnv
benchmark_times = {
    "dmu05": 0.16,
    "dmu10": 0.21,
    "dmu15": 0.41,
    "dmu20": 0.46,
    "dmu25": 0.57,
    "dmu30": 0.9,
    "dmu35": 0.88,
    "dmu40": 1.2,
    "dmu45": 0.16,
    "dmu50": 0.22,
    "dmu55": 0.35,
    "dmu60": 0.46,
    "dmu65": 0.59,
    "dmu70": 0.82,
    "dmu75": 0.88,
    "dmu80": 1.2,
    "la05": 0.01,
    "la10": 0.02,
    "la15": 0.04,
    "la20": 0.03,
    "la25": 0.06,
    "la30": 0.1,
    "la35": 0.21,
    "la40": 0.09,
    "orb10": 0.03,
    "swv05": 0.1,
    "swv10": 0.16,
    "swv20": 0.55,
    "ta10": 0.1,
    "ta20": 0.15,
    "ta30": 0.21,
    "ta40": 0.33,
    "ta50": 0.45,
    "ta60": 0.88,
    "ta70": 1.2,
    "ta80": 4.6,
    "lta_j100_m100_10" : 90.0,
    "lta_j1000_m10_10": 90.0,
}

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
            return f"{mean_value:>3.0f}{unit}"

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

    for unit in ("s ", "ms", "µs"):
        if mean_value >= 1:
            break
        
        mean_value *= 1000
        std_val *= 1000

    else:
        unit = "ns"

    if std_val > 0:
        return f"{mean_value:.2f} ± {std_val:.2f} {unit}"

    return f"{mean_value:6.2f} {unit}"


def format_percentage(value: list[float]) -> str:
    mean_value = mean(value)
    std_val = std(value)

    if std_val > 0:
        return f"({100*mean_value:.2f} ± {100*std_val:.2f})%"

    return f"{100*mean_value:.2f}%"

def format_stage_time(
    stage_times: list[float],
    total_time: float,
) -> str:
    stage_statistics = format_time(stage_times)

    pct = mean(stage_times) / total_time if total_time > 0 else 0.0

    percentage = format_percentage([pct])

    str_size = len(stage_statistics)

    if str_size < 10:
        spacing = " " * (10 - str_size)
    
    else:
        spacing = " " * (18 - len(stage_statistics))

    idx = int(len(COLORMAP) * pct)
    idx = max(0, min(len(COLORMAP) - 1, idx))

    return stage_statistics + to_ansi_color(COLORMAP[idx]) + f"{spacing}({percentage})" + RESET

class RunResult:
    MIN_SPEEDUP = 4
    "Speedup required to be considered a significant improvement. Maps to colormap[0]."

    MAX_SPEEDUP = 20
    "Speedup considered a very significant improvement. Maps to colormap[-1]."

    # Static values
    instance_names: list[str]
    entries: dict[str, str]
    n_tasks: dict[str, int]
    n_events: dict[str, int]

    # Dynamic values
    initialization_times: dict[str, list[float]]
    reset_times: dict[str, list[float]]
    pdr_times: dict[str, list[float]]
    step_times: dict[str, list[float]]
    simulation_times: dict[str, list[float]]

    def __init__(self) -> None:
        self.current_instance: str | None = None
        self.current_run = 0

        self.instance_names = []

        self.entries = {}
        self.n_tasks = {}
        self.n_events = {}

        self.initialization_times = {}
        self.reset_times = {}
        self.pdr_times = {}
        self.step_times = {}
        self.simulation_times = {}

    def start_instance(
        self,
        instance_name: str,
        env: SchedulingEnv,
    ) -> None:
        self.current_instance = instance_name
        self.instance_names.append(instance_name)
        self.n_tasks[instance_name] = env.state.n_tasks
        self.n_events[instance_name] = env.event_count
        self.entries[instance_name] = env.get_entry().replace("|", "/")

        self.current_run = 0
        self.initialization_times[instance_name] = []
        self.reset_times[instance_name] = []
        self.pdr_times[instance_name] = []
        self.step_times[instance_name] = []
        self.simulation_times[instance_name] = []
    

    def log_run(
        self,
        setup_time: float,
        reset_time: float,
        pdr_time: float,
        step_time: float,
        simulation_time: float,
        
    ) -> None:
        if self.current_instance is None:
            raise ValueError("No instance started. Call start_instance() before logging runs.")

        instance_name = self.current_instance

        self.initialization_times[instance_name].append(setup_time)
        self.reset_times[instance_name].append(reset_time)
        self.pdr_times[instance_name].append(pdr_time)
        self.step_times[instance_name].append(step_time)
        self.simulation_times[instance_name].append(simulation_time)

    def print_results(self, full: bool) -> None:
        table = PrettyTable([
            "Instance",
            "Entry",
            "Tasks",
            "Simulation time",
        ] + ([
            "Initialization time",
            "Reset time",
            "PDR time",
            "Step time",
        ] if full else []) + [
            "Time per task",
            "Events",
            "Events/task",
            "Events/sec",
            "Speedup"
        ])
        table.set_style(TableStyle.MARKDOWN)

        for instance_name in self.instance_names:
            entry = self.entries[instance_name]
            tasks = format_big_number([self.n_tasks[instance_name]])
            simulation_time = format_time(self.simulation_times[instance_name])

            total_time = mean(self.simulation_times[instance_name])
    
            initialization_time = format_stage_time(self.initialization_times[instance_name], total_time)
            reset_time = format_stage_time(self.reset_times[instance_name], total_time)
            pdr_time = format_stage_time(self.pdr_times[instance_name], total_time)
            step_time = format_stage_time(self.step_times[instance_name], total_time)

            time_per_task = format_time(
                [t / self.n_tasks[instance_name] for t in self.simulation_times[instance_name]]
            )
            events = format_big_number([self.n_events[instance_name]])
            events_per_task = format_big_number([self.n_events[instance_name] / self.n_tasks[instance_name]])
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
                entry,
                tasks,
                simulation_time,
            ] + ([
                initialization_time,
                reset_time,
                pdr_time,
                step_time,
            ] if full else []) + [
                time_per_task,
                events,
                events_per_task,
                events_per_sec,
                speedup
            ])

        print("\n")
        print(table, flush=True)

    def plot_results(self) -> None:
        from matplotlib.axes import Axes
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy import stats
        from statistics import mean

        sns.set_theme(style="whitegrid", palette="Set2")

        ax: Sequence[Axes]
        fig, ax = plt.subplots(
            ncols=2,
            figsize=(18, 4),
            width_ratios=[2, 1],
            constrained_layout=True,
        )

        stages = ["initialization", "reset", "pdr", "step"]

        # Sort instances by total time descending
        totals = [
            sum(mean(self.__dict__[f"{stage}_times"][i]) for stage in stages)
            for i in self.instance_names
        ]
        sorted_pairs = sorted(zip(totals, self.instance_names), reverse=True)
        sorted_totals, sorted_instances = zip(*sorted_pairs)

        bottom = [0.0] * len(sorted_instances)
        for stage in stages:
            stage_times = [
                mean(self.__dict__[f"{stage}_times"][instance])
                for instance in sorted_instances
            ]

            ax[0].bar(
                sorted_instances,
                stage_times,
                label=stage,
                linewidth=0,
                bottom=bottom,
            )

            bottom = [b + t for b, t in zip(bottom, stage_times)]

        for i, (_, total) in enumerate(zip(sorted_instances, sorted_totals)):
            ax[0].text(
                i,
                total + 0.01 * max(sorted_totals),
                f"{1000 * total:.0f} ms" if total >= 0.001 else f"{total * 1e6:.0f} µs",
                ha="center",
                va="bottom",
                fontsize=5,
                # rotation=90,
            )

        ax[0].margins(x=0.01, y=0.1)
        ax[0].set_title("Time per stage for each instance")
        ax[0].set_xlabel("Instance")
        ax[0].set_ylabel("Time (s)")

        ax[0].legend(title="Stage", loc="upper right")
        ax[0].set_xticks(range(len(sorted_instances)))
        ax[0].set_xticklabels(sorted_instances, rotation=90, fontsize=6)

        # --- Right plot ---
        fit_color = "#2563EB"

        task_numbers = [self.n_tasks[instance] for instance in self.instance_names]
        perf = [mean(self.simulation_times[instance]) for instance in self.instance_names]

        slope, intercept, *_, std_err = stats.linregress(
            np.log(task_numbers), np.log(perf)
        )

        alpha = 0.05
        t_val = stats.t.ppf(1 - alpha / 2, len(task_numbers) - 2)
        exp_conf_interval = t_val * std_err

        x = np.logspace(
            np.log10(min(task_numbers)) - 0.05,
            np.log10(max(task_numbers)) + 0.05,
            100,
        )

        assert isinstance(intercept, float)

        ax[1].plot(
            x,
            exp(intercept) * x**slope,
            color=fit_color,
            label=f"Linear Fit (k={slope:.2f} ± {exp_conf_interval:.2f})",
            zorder=1,
        )

        ax[1].fill_between(
            x,
            exp(intercept) * x ** (slope - exp_conf_interval),
            exp(intercept) * x ** (slope + exp_conf_interval),
            color=fit_color,
            alpha=0.15,
            label="95% Confidence Interval",
            zorder=0,
        )

        perf_err = [std(self.simulation_times[instance]) for instance in self.instance_names]

        yerr_lower = [p - p / np.exp(e / p) for p, e in zip(perf, perf_err)]
        yerr_upper = [p * np.exp(e / p) - p for p, e in zip(perf, perf_err)]

        ax[1].errorbar(
            task_numbers,
            perf,
            yerr=[yerr_lower, yerr_upper] if self.current_run > 1 else None,
            color="black",
            fmt="o",
            ecolor="black",
            elinewidth=2,
            capsize=4,
            label="Measured Times",
            zorder=2,
        )

        ax[1].loglog()
        ax[1].set_title("Average time vs Number of Tasks")
        ax[1].set_xlabel("Number of Tasks")
        ax[1].set_ylabel("Average time (s)")
        ax[1].legend()

        fig.suptitle("Speed Benchmark Report", fontweight="bold")

        fig.savefig(ROOT / "report.pdf", dpi=300)
        plt.show()


def test_memory(pdr: PDR_NAMES, dynamic: bool, quiet: bool) -> None:
    """Run a single iteration per instance with tracemalloc to measure peak memory usage."""
    import tracemalloc

    spt_agent = PDRS[pdr]()

    columns = ["Instance", "Tasks", "Peak Memory", "Memory/Task"]
    table = PrettyTable(columns)
    table.set_style(TableStyle.MARKDOWN)

    if not quiet:
        print(f"Running memory benchmark", end="")

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

        instance_path = ROOT / "instances/jobshop" / f"{instance_name}.txt"

        instance, _ = read_jsp_instance(instance_path)

        tracemalloc.start()
        env = SchedulingEnv(JobShopSetup())
        env.set_instance(instance)

        obs, _ = env.reset()

        if dynamic:
            done = False
            while not done:
                single_action = spt_agent(obs)
                obs, _, done, _, _ = env.step(single_action)
        else:
            action = spt_agent.ranking(obs)
            env.step(action)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        n_tasks = env.state.n_tasks
        per_task = peak_mem / n_tasks if n_tasks > 0 else 0

        table.add_row([
            instance_name,
            format_big_number([n_tasks]),
            format_memory(peak_mem),
            format_memory(per_task),
        ])

        del env

    print("\n")
    print(table, flush=True)

def print_header() -> None:
    print(f"cpscheduler v{__version__}")
    print(f"{ok_fail(__compiled__)} compiled")
    
    instance_present = (ROOT / "instances/jobshop").exists()
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
    gym: Annotated[bool, arg(aliases=("-g",))] = False,
    pdr: Annotated[PDR_NAMES, arg(aliases=("-p",))] = "spt",
    quiet: Annotated[bool, arg(aliases=("-q",))] = False,
    plot: bool = False,
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

    gym: bool
        If True, run the benchmark using the Gym interface.

    pdr: PDR_NAMES
        The priority dispatching rule to use for the benchmark.

    quiet: bool
        If True, suppress the output of the benchmark results.

    numpy: bool
        If True, allow the use of NumPy in the agent's decision making. Disabling
        this can help isolate the time spent on the environment itself.

    dynamic: bool
        If True, run the benchmark in dynamic mode, where the agent makes a decision
        at each step. If False, the agent makes a single decision for the entire schedule.

    plot: bool
        If True, plot the benchmark results.
    """
    if not quiet:
        print_header()

    if memory:
        print("Running bechmark: memory usage")

        test_memory(pdr, dynamic, quiet)
        return


    print("Running bechmark: time")
    result = RunResult()

    agent = PDRS[pdr]()

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


        instance_path = ROOT / "instances/jobshop" / f"{instance_name}.txt"
        instance, _ = read_jsp_instance(instance_path)
        gc.collect()
        gc.freeze()

        env: SchedulingEnv | SchedulingEnvGym
        for i in range(n_runs):
            gc.disable()

            global_tick = perf_counter()
            if gym:
                env = SchedulingEnvGym(JobShopSetup(), objective=Makespan(), instance=instance)
            
            else:
                env = SchedulingEnv(JobShopSetup(), objective=Makespan(), instance=instance)

            initialization_times = perf_counter() - global_tick

            tick = perf_counter()
            obs, _ = env.reset()
            reset_time = perf_counter() - tick

            if dynamic:
                pdr_time = 0.
                step_time = 0.

                done = False
                while not done:
                    tick = perf_counter()
                    single_action = agent(obs)
                    pdr_time += perf_counter() - tick

                    assert single_action is not None

                    tick = perf_counter()
                    obs, _, done, *_ = env.step(single_action)
                    step_time += perf_counter() - tick
            
            else:
                tick = perf_counter()
                action = agent.ranking(obs)
                pdr_time = perf_counter() - tick

                tick = perf_counter()
                env.step(action)
                step_time = perf_counter() - tick

            simulation_time = perf_counter() - global_tick

            if i == 0:
                _env = env.core if isinstance(env, SchedulingEnvGym) else env

                result.start_instance(instance_name, _env)


            result.log_run(
                initialization_times,
                reset_time,
                pdr_time,
                step_time,
                simulation_time,
            )

            del env
            gc.enable()
            gc.collect()

        gc.collect()
        gc.unfreeze()
    result.print_results(full=full)

    if plot:
        result.plot_results()


if __name__ == "__main__":
    tyro.cli(
        run_cli,
        description="Test the speed of the Shortest Processing Time heuristic on various job shop instances.",
    )
