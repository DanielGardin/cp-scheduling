"""Benchmark to measure peak memory usage of the scheduling environment."""

from typing import Annotated

import tyro
from _common import (
    PDR_NAMES,
    PDRS,
    ROOT,
    format_big_number,
    format_memory,
    print_header,
)
from prettytable import PrettyTable, TableStyle
from tyro.conf import arg

from cpscheduler.environment import (
    JobShopSetup,
    Makespan,
    SchedulingEnv,
)
from cpscheduler.instances.jobshop import read_jsp_instance

benchmark_instances = [
    "dmu05",
    "dmu10",
    "dmu15",
    "dmu20",
    "dmu25",
    "dmu30",
    "dmu35",
    "dmu40",
    "dmu45",
    "dmu50",
    "dmu55",
    "dmu60",
    "dmu65",
    "dmu70",
    "dmu75",
    "dmu80",
    "la05",
    "la10",
    "la15",
    "la20",
    "la25",
    "la30",
    "la35",
    "la40",
    "orb10",
    "swv05",
    "swv10",
    "swv20",
    "ta10",
    "ta20",
    "ta30",
    "ta40",
    "ta50",
    "ta60",
    "ta70",
    "ta80",
    "lta_j100_m100_10",
    # "lta_j1000_m10_10",
]


def run_cli(
    pdr: Annotated[PDR_NAMES, arg(aliases=("-p",))] = "spt",
    quiet: Annotated[bool, arg(aliases=("-q",))] = False,
    dynamic: bool = False,
) -> None:
    """Run a single iteration per instance with tracemalloc to measure peak memory usage."""
    import tracemalloc

    spt_agent = PDRS[pdr]()

    columns = ["Instance", "Tasks", "Peak Memory", "Memory/Task"]
    table = PrettyTable(columns)
    table.set_style(TableStyle.MARKDOWN)

    if not quiet:
        print_header()
        print("Running bechmark: memory usage", end="")

    dots = 0
    for instance_name in benchmark_instances:
        if not quiet:
            if dots < 3:
                print(".", end="", flush=True)
                dots += 1

            else:
                print(
                    f"\r{' ' * 100}",
                    end="\rRunning bechmark: memory usage",
                    flush=True,
                )
                dots = 0

        instance_path = ROOT / "instances/jobshop" / f"{instance_name}.txt"

        instance, _ = read_jsp_instance(instance_path)

        tracemalloc.start()
        env = SchedulingEnv(
            JobShopSetup(), objective=Makespan(), instance=instance
        )

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

        table.add_row(
            [
                instance_name,
                format_big_number([n_tasks]),
                format_memory(peak_mem),
                format_memory(per_task),
            ]
        )

        del env

    if not quiet:
        print()

    print(table, flush=True)


if __name__ == "__main__":
    tyro.cli(
        run_cli,
        description="Run a benchmark to measure peak memory usage of the scheduling environment.",
    )
