from pathlib import Path

from typing import Any, Literal, Annotated
from collections.abc import Callable
from torch.types import Device

from pandas import DataFrame

import pandas as pd

from gymnasium.vector import AsyncVectorEnv

import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from gymnasium import Env, make_vec

from dataclasses import dataclass

import tyro
from tyro.conf import arg, Positional

from cpscheduler import SingleMachineSetup, WeightedTardiness, TotalTardiness
from cpscheduler.environment.metrics import ReferenceScheduleMetrics
from cpscheduler.gym import (
    SchedulingEnvGym, PermutationActionWrapper, InstancePoolWrapper, ArrayObservationWrapper
)
import cpscheduler.rl as rl

INSTANCE_PATH = Path("data/customers/wt40_50_0.5.pkl")
DATASET_PATH = Path("data/customers/wt40_50_0.5_dataset.pkl")

features = ["processing_time", "due_date", "customer"]

@dataclass
class Config:
    dataset: Positional[Path] = DATASET_PATH
    "Path to the customer dataset."

    instance: Positional[Path] = INSTANCE_PATH
    "Path to the instance file."

    seed: int | None = None
    "Random seed for reproducibility."

    log_dir: str = "logs"
    "Directory to save logs and model checkpoints."

    device: Annotated[Device, arg(aliases=("-d",))] = "auto"
    "Device to run the training on, e.g., 'cpu' or 'cuda'."

    quiet: Annotated[bool, arg(aliases=("-q",))] = False
    "If True, suppresses output during training."

    frac_train: float = 1.0
    "Fraction of the dataset to use for training."

    hidden_size: Annotated[int, arg(aliases=("-s",))] = 64
    "Size of the hidden layers in the neural network."

    n_layers: Annotated[int, arg(aliases=("-l",))] = 2
    "Number of layers in the neural network."

    dropout: float = 0.0
    "Dropout rate for the neural network."

    lr: float = 1e-5
    "Learning rate for the optimizer."

    weight_decay: float = 0.0
    "Weight decay for the optimizer."

    n_envs: Annotated[int, arg(aliases=("-e",))] = 64
    "Number of parallel environments to run."

    baseline: Literal['mean', 'greedy', 'none'] = "greedy" 
    "Baseline method to use for the policy gradient."

    norm_reward: bool = True
    "If True, normalizes the reward during training with the running standard deviation."

    steps: Annotated[int, arg(aliases=("-n",))] = 100_000
    "Number of steps to perform during training."

    steps_per_update: int = 4
    "Number of steps to take per update."

    batch_size: Annotated[int, arg(aliases=("-b",))] = 1
    "Batch size for training."

    validation_freq: Annotated[int, arg(aliases=("-v",))] = 1
    "Frequency of validation during training."


args = tyro.cli(Config)

if args.seed is not None:
    rl.utils.set_seed(args.seed)

device = rl.utils.get_device(args.device)

list_instances: list[DataFrame]
instance_info: DataFrame
customer_information: DataFrame

list_instances, instance_info, customer_information = pd.read_pickle(args.instance)
train_dataset: list[DataFrame] = pd.read_pickle(args.dataset)

ref = ReferenceScheduleMetrics("BPolicy start")

def make_env(instance: DataFrame) -> Callable[[], Env[Any, Any]]:
    def inner() -> Env[Any, Any]:
        env: Env[Any, Any] = SchedulingEnvGym(
            machine_setup=SingleMachineSetup(),
            objective=WeightedTardiness(),
            instance_config={"instance": instance},
            metrics={
                "total_tardiness": TotalTardiness(),
                "order_preservation": ref.order_preservation,
                "accuracy": ref.hamming_accuracy,
                "kendall_tau": ref.kendall_tau,
            }
        )
        env = ArrayObservationWrapper(env, features)
        env = PermutationActionWrapper(env)
        return env

    return inner

val_envs = AsyncVectorEnv(
    [
        make_env(instance)
        for i, instance in enumerate(list_instances)
    ]
)

envs = make_vec(
    "Scheduling-v0",
    num_envs=args.n_envs,
    vectorization_mode="async",
    wrappers=[
        lambda env: ArrayObservationWrapper(env, features),
        lambda env: InstancePoolWrapper(env, train_dataset),
        PermutationActionWrapper,
    ],
    machine_setup=SingleMachineSetup(),
    objective=TotalTardiness(),
    metrics={
        "order_preservation": ref.order_preservation,
        "accuracy": ref.hamming_accuracy,
        "kendall_tau": ref.kendall_tau,
    }
)

val_obs = torch.cat(
    [
        torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(0)
        for df in list_instances
    ]
).to(device)

dataset = torch.cat(
    [
        torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(0)
        for df in train_dataset
    ]
).to(device)

if args.frac_train < 1:
    indices = torch.randperm(int(dataset.shape[0] * args.frac_train)).to(device)

    dataset = dataset[indices]

preprocessor = rl.preprocessor.TabularPreprocessor(
    categorical_indices=[2],
    numerical_indices=[0, 1],
    categorical_embedding_dim=8,
)

preprocessor.fit(dataset)

scorer = nn.Sequential(
    preprocessor,
    rl.network.MLP(
        input_dim=preprocessor.output_dim,
        output_dim=1,
        hidden_dims= [args.hidden_size] * args.n_layers,
        dropout=args.dropout,
    )
)

policy = rl.policies.PlackettLucePolicy(scorer).to(device)
# Compile the policy

optimizer = Adam(
    policy.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

policy.compile()

class SchedulingREINFORCE(rl.online.Reinforce):
    def validate(self) -> dict[str, Any]:
        val_envs.reset()

        with torch.no_grad():
            action = self.policy.greedy(val_obs).cpu().numpy()

        _, returns, *_, info = val_envs.step(action)

        del info["current_time"]
        del info["n_queries"]

        return {
            key: value
            for key, value in info.items()
            if not key.startswith('_')
        } | {
            "optimal_gap": (-np.sum(returns) / np.sum(instance_info["optimal"]) - 1).item()
        }

obs_shape: tuple[int, ...] = (
    envs.single_observation_space.shape
    if envs.single_observation_space.shape is not None 
    else (1,)
)
act_shape = obs_shape[:1]

with SchedulingREINFORCE(
    obs_shape,
    act_shape,
    envs,
    policy,
    optimizer,
    args.baseline,
    norm_returns=args.norm_reward,
    device=device,
) as algo:
    algo.learn(
        num_updates      = args.steps,
        global_steps= args.steps,
        steps_per_update = args.steps_per_update,
        batch_size       = args.batch_size,
        validation_freq  = args.validation_freq,
        log_dir=args.log_dir,
        quiet=args.quiet,
    )
