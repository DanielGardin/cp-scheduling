from pathlib import Path
from typing import Annotated

from pandas import DataFrame

import pandas as pd

import numpy as np

import torch

from tqdm import tqdm

from cpscheduler import SchedulingEnv, SingleMachineSetup, WeightedTardiness
from cpscheduler.solver import PulpSolver

from cpscheduler.rl.policies import PlackettLucePolicy

list_instances: list[DataFrame]
instance_info: DataFrame
customer_information: DataFrame

list_instances, instance_info, customer_information = pd.read_pickle("data/customers/wt40_50_0.5.pkl")

env = SchedulingEnv(
    machine_setup=SingleMachineSetup(),
    objective=WeightedTardiness(),
)


policy: PlackettLucePolicy = torch.load("models/pl_behavior_cloning/policy.pkl", weights_only=False)


data = torch.cat(
    [
        torch.tensor(df[["processing_time", "due_date", "customer"]].values, dtype=torch.float32).unsqueeze(0)
        for df in list_instances
    ]
)

max_gap = 0.01

actions = policy.greedy(data)
optimal_actions = np.zeros((len(list_instances), len(list_instances[0])), dtype=int)

with tqdm(list_instances) as pbar:
    for i, instance in enumerate(pbar):
        optimal_value = float(instance_info.loc[i, "optimal"]) # type: ignore
        assert isinstance(optimal_value, (int, float))

        pbar.set_description(f"Instance {i} - Optimal Value: {optimal_value}")

        env.reset(options={"instance": instance})

        solver = PulpSolver(env)
        solver.warm_start([("execute", a.item()) for a in actions[i]])

        sol, returns, optimal = solver.solve(
            solver_tag="CPLEX_CMD",
            time_limit=60,
            # quiet=True,
            options=[
                f"set mip limits lowerobjstop {optimal_value * 1.001}",
            ]
        )

        pbar.set_postfix(
            solution_value=returns,
            gap=(
                returns / optimal_value - 1
                if optimal_value > 0 else
                0 if returns == 0 else
                float("inf")
            )
        )

        optimal_actions[i] = [a[1] for a in sol] # type: ignore


np.save("optimal_actions.npy", optimal_actions)