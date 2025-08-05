from pathlib import Path

from typing import Any, Annotated
from collections.abc import Iterable
from pandas import DataFrame
from numpy.typing import NDArray

import random

import numpy as np
import pandas as pd

from dataclasses import dataclass

from tqdm import tqdm

from natsort import natsorted

import tyro
from tyro.conf import Positional, arg

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

from cpscheduler.environment import SchedulingEnv, SingleMachineSetup, WeightedTardiness, Objective
from cpscheduler.instances import read_smtwt_instance

from cpscheduler.solver import PulpSolver
from cpscheduler.heuristics import (
    PriorityDispatchingRule,
    # ShortestProcessingTime,
    WeightedShortestProcessingTime,
    # EarliestDueDate,
    ModifiedDueDate,
    ApparentTardinessCost,
    # MinimumSlackTime,
    TrafficPriority,
    CostOverTime,
    # CriticalRatio,
    # RandomPriority
)
import logging.config

import logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PWD = Path(__file__).parent
ROOT = Path(__file__).parent.parent.parent

pdrs: dict[str, PriorityDispatchingRule] = {
    "WSPT": WeightedShortestProcessingTime(strict=True),
    "MDD" : ModifiedDueDate(strict=True),
    "WMDD": ModifiedDueDate(weight_label="weight", strict=True),
    "COverT": CostOverTime(strict=True),
    "ATC": ApparentTardinessCost(5, strict=True),
    "TP" : TrafficPriority(strict=True),
}

def mean_point(arr: NDArray[Any]) -> NDArray[Any]:
    a = np.insert(arr, 0, 0)

    return a[:-1] + np.diff(a) / 2

def common_instance_name(instance_names: list[str]) -> str:
    "Get the substring of the common instance name in the pattern <instance_name>_<id>.<ext>"
    max_idx = min(len(name) for name in instance_names)

    common_name = instance_names[0]
    for name in instance_names[1:]:
        for idx in range(max_idx):
            if name[idx] != common_name[idx]:
                max_idx = idx
                break
    
    return common_name[:max_idx].strip(" _\n\t.")


def read_instances(paths: list[str]) -> tuple[DataFrame, list[float]]:
    n_instances = len(paths)
    if n_instances == 0:
        logger.error("No instances were added. Usage: python customer.py <paths> [...]")
    
    else:
        logger.info(f"Processing {n_instances} instances file{'s' if n_instances > 1 else ''}")

    base_instances: list[DataFrame] = []
    optimal_bounds = [0. for _ in range(n_instances)]
    for i, path in enumerate(natsorted(paths)):
        instance, metadata = read_smtwt_instance(path)

        base_instances.append(
            pd.DataFrame(instance).assign(instance_id=i)
        )
        optimal_bounds[i] = float(metadata["Objective UB"])

    return pd.concat(base_instances, ignore_index=True), optimal_bounds


def customer_scheduling_instance(
    base_instance: DataFrame,
    n_customers: int,
    sort_noise: float = 0.0,
    *,
    weight_feature: str = "weight",
    customer_feature: str = "customer",
    seed: int | None = None,
) -> tuple[list[DataFrame], DataFrame]:
    """
    Adapt a set of scheduling instances to a customer scheduling instance.
    The customer scheduling instance is a scheduling instance where the jobs are
    owned by customers, who have different weights.

    Given an instance of a α | β | Σ wj h(.) scheduling problem, we create a new customer
    feature such that jobs with equal weights are assigned to customers, stabilishing a
    relation between the customer and the job weights.

    The assumptions are:
    - The customer presence is sampled uniformly from a simplex.
    - Customer weights are correlated to the number of jobs assigned to them (this correlation
      can be controlled by the `sort_noise` parameter).


    Parameters
    ----------
    base_instance : DataFrame
        The base instance to adapt.

    """
    rng = np.random.default_rng(seed)

    weights = np.sort(base_instance[weight_feature].unique())

    weights_distribution = (
        base_instance[weight_feature].value_counts().sort_index().to_numpy()
    )
    weights_distribution = weights_distribution / np.sum(weights_distribution)

    # Customer presences are sampled uniformly from a simplex
    customer_presence = rng.dirichlet(np.ones(n_customers))

    customers_order = np.argsort(
        np.log(customer_presence) + sort_noise * rng.gumbel(0, 1, n_customers)
    )

    cumulative_presence = mean_point(np.cumsum(customer_presence[customers_order]))

    assignment = np.searchsorted(cumulative_presence, np.cumsum(weights_distribution))
    n_customers_per_weight = np.diff(np.insert(assignment, 0, 0))

    # Check if any weight bin has no customers assigned and tweak the boundaries
    for i in range(len(n_customers_per_weight) - 1, 0, -1):
        current = n_customers_per_weight[i]

        if current <= 0:
            required = 1 - current

            n_customers_per_weight[i] += required
            n_customers_per_weight[i - 1] -= required

    customer_information = DataFrame(
        {weight_feature: 0, "n_jobs": 0, "presence": customer_presence}
    )

    idx = 0
    for weight_bin, weight in enumerate(weights):
        n_jobs = len(base_instance[base_instance[weight_feature] == weight])
        n_customers_in_bin = n_customers_per_weight[weight_bin]

        customers = customers_order[idx : idx + n_customers_in_bin]
        bin_presence = customer_presence[customers] / customer_presence[customers].sum()

        jobs_distribution = (
            rng.multinomial(n_jobs - n_customers_in_bin, bin_presence) + 1
        )

        idx += int(n_customers_in_bin)

        customer_information.loc[customers.tolist(), weight_feature] = weight
        customer_information.loc[customers.tolist(), "n_jobs"] = jobs_distribution

        base_instance.loc[base_instance[weight_feature] == weight, customer_feature] = (
            rng.permutation(np.repeat(customers, jobs_distribution))
        )

    weight_norm = customer_information[weight_feature].sum()

    base_instance[weight_feature] = base_instance[weight_feature] / weight_norm

    customer_information[f"{weight_feature}_original"] = customer_information[weight_feature]
    customer_information[weight_feature] = customer_information[weight_feature] / weight_norm

    base_instance[customer_feature] = base_instance[customer_feature].astype(int)

    list_instances = [
        instance.drop(columns=["instance_id"])
        for _, instance in base_instance.groupby("instance_id")
    ]

    return list_instances, customer_information


def plot_customer_allocation(
    customer_information: DataFrame,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    n_weights = customer_information["weight"].nunique()

    palette = sns.color_palette("rocket_r", n_colors=n_weights)
    
    binned_info = customer_information.groupby("weight")

    start = 0
    mean_points: list[float] = []
    weights: list[float] = []
    for i, (weight, group) in enumerate(binned_info):
        original_weight = group["weight_original"].iloc[0]

        ax.bar(
            range(start, start + len(group)),
            group["presence"].sort_values(),
            color=palette[i],
            alpha=0.9,
            label=f"Weight {original_weight}",
        )

        ax.fill_between(
            range(start, start + len(group)),
            group["presence"].sort_values(),
            color=palette[i],
            alpha=0.3,
        )

        weights.append(original_weight)

        mean_points.append(start + (len(group) - 1) / 2)
        start += len(group)


    ax.set_xlabel("Weights")
    ax.set_ylabel("Customer Presence")
    ax.set_title("Customer presence per weight bin")

    ax.set_xticks(mean_points, map(str, weights))

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1, .9),
        title="Weights",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(save_path) if save_path else plt.show()


if __name__ == "__main__":
    @dataclass
    class Config:
        instances: Positional[list[str]]
        "Paths of instances to adapt, including * wildcard."

        seed: int | None = None
        "Random seed for reproducibility."

        n_customers: Annotated[int, arg(aliases=('-n',))] = 50
        "Number of customers to assign jobs to."

        sort_noise: float = 0.5
        "How much noise is inserted into the customer assignment."

        plots: Annotated[bool, arg(aliases=('-p',))] = False
        "Generate plots for the instance generation."

        solver: Annotated[str, arg(aliases=('-s',))] = "CPLEX_CMD"
        "Solver to use during action collection. Set to `none` to disable solver usage."

        timelimit: Annotated[int, arg(aliases=('-t',))] = 30
        "Time limit for the solver in seconds."

    args = tyro.cli(Config)

    rng = random.Random(args.seed)

    choices = [*pdrs.keys(), "Optimal"] if args.solver.lower() != "none" else [*pdrs.keys()]

    base_instance, optimal_bounds = read_instances(args.instances)

    list_instances, customer_information = customer_scheduling_instance(
        base_instance,
        n_customers=args.n_customers,
        sort_noise=args.sort_noise,
        seed=args.seed,
    )

    weight_norm = float(customer_information["weight"].sum() / customer_information["weight_original"].sum())
    optimal_bounds = [bound * weight_norm for bound in optimal_bounds]

    if args.plots:
        plot_customer_allocation(customer_information, PWD / "customer_allocation.pdf")

    env = SchedulingEnv(
        machine_setup=SingleMachineSetup(),
        objective=WeightedTardiness(),
    )

    behavior_reward = [0. for _ in range(len(list_instances))]
    with tqdm(
        list_instances,
        desc="Scheduling instances",
        unit="instance",
        disable=not logger.isEnabledFor(logging.INFO),
    ) as pbar:
        pbar.set_description("Collecting behavior data")
        pbar.set_postfix_str("")

        for i, instance in enumerate(pbar):
            env.set_instance(instance)
            obs, info = env.reset()

            strategy = rng.choice(choices)
            pbar.set_postfix_str(f"Strategy: {strategy}")


            if strategy == "Optimal":
                solver = PulpSolver(env)
                action, _, _ = solver.solve(args.solver, quiet=True, time_limit=args.timelimit)

            else:
                pdr = pdrs[strategy]
                action = pdr.sample(obs, info["current_time"], target_prob=0.9, seed=i)

            new_obs, reward, done, truncated, info = env.step(action)

            instance["BPolicy start"] = [task.get_start() for task in env.tasks]
            behavior_reward[i] = -reward if env.objective.minimize else reward

    logger.info(
        "Scheduling instances completed. Generating statistics:"
    )
    logger.info(f"Cumulative gap: {sum(behavior_reward) / sum(optimal_bounds) - 1:.2%}")

    instance_info = pd.DataFrame(
        {
            "optimal": optimal_bounds,
            "behavior": behavior_reward
        }
    )

    save_instances = list_instances, instance_info, customer_information

    instances_name = common_instance_name(
        [Path(path).stem for path in args.instances]
    )
    if not instances_name:
        instances_name = "customer_instances"

    pd.to_pickle(
        save_instances,
        PWD / f"{instances_name}_{args.n_customers}_{args.sort_noise}.pkl"
    )