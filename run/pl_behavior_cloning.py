from pathlib import Path

from typing import Any, Annotated
from collections.abc import Callable
from torch.types import Device
from numpy.typing import NDArray

from pandas import DataFrame

import pandas as pd

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from gymnasium import Env, make_vec

from dataclasses import dataclass, asdict

import tyro
from tyro.conf import arg, Positional

from cpscheduler import SingleMachineSetup, WeightedTardiness, TotalTardiness, Objective
from cpscheduler.environment.metrics import ReferenceScheduleMetrics
from cpscheduler.gym import (
    SchedulingEnvGym, PermutationActionWrapper, ArrayObservationWrapper
)
import cpscheduler.rl as rl

INSTANCE_PATH = "data/customers/wt40_50_0.5.pkl"
DATASET_PATH = "data/customers/wt40_50_0.5_dataset.pkl"

features = ["processing_time", "due_date", "customer"]

@dataclass
class Config:
    dataset: Positional[str] = DATASET_PATH
    "Path to the customer dataset."

    instance: Positional[str] = INSTANCE_PATH
    "Path to the instance file."

    log_dir: str = "logs"
    "Directory to save logs and model checkpoints."

    frac_train: float = 1
    "Fraction of the dataset to use for training."

    seed: int | None = None
    "Random seed for reproducibility."

    device: Annotated[Device, arg(aliases=("-d",))] = "auto"
    "Device to run the training on, e.g., 'cpu' or 'cuda'."

    quiet: Annotated[bool, arg(aliases=("-q",))] = False
    "If True, suppresses output during training."

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

    steps: Annotated[int, arg(aliases=("-n",))] = 100_000
    "Number of steps to perform during training."

    steps_per_update: int = 1
    "Number of steps to take per update."

    batch_size: Annotated[int, arg(aliases=("-b",))] = 1
    "Batch size for training."

    validation_freq: Annotated[int, arg(aliases=("-v",))] = 1
    "Frequency of validation during training."


args = tyro.cli(Config)

if args.seed is not None:
    rl.utils.set_seed(args.seed)

device = rl.utils.get_device(args.device)
args.device = device

list_instances: list[DataFrame]
instance_info: DataFrame
customer_information: DataFrame

list_instances, instance_info, customer_information = pd.read_pickle(Path(args.instance))
train_dataset: list[DataFrame] = pd.read_pickle(Path(args.dataset))

ref = ReferenceScheduleMetrics("BPolicy start")

def make_env(instance: DataFrame) -> Callable[[], Env[Any, Any]]:
    def inner() -> Env[Any, Any]:
        env: Env[Any, Any] = SchedulingEnvGym(
            machine_setup=SingleMachineSetup(),
            objective=WeightedTardiness() if "weight" in instance.columns else Objective(),
            instance_config={"instance": instance},
            metrics={
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
        for instance in list_instances
    ]
)

envs = SyncVectorEnv(
    [
        make_env(instance)
        for instance in train_dataset
    ]
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

actions = torch.cat(
    [
        torch.tensor(
            df["BPolicy start"].values,
            dtype=torch.float32
        ).unsqueeze(0).argsort()
        for df in train_dataset
    ]
).to(device)

if args.frac_train < 1:
    indices = torch.randperm(int(dataset.shape[0] * args.frac_train)).to(device)

    dataset = dataset[indices]
    actions = actions[indices]

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

optimizer = Adam(
    policy.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

policy.compile()

class SchedulingBC(rl.offline.BehaviorCloning):
    def on_epoch_end(self) -> dict[str, Any]:
        obs: NDArray[Any]

        obs, _ = envs.reset()
        
        with torch.no_grad():
            action = self.policy.greedy(torch.tensor(obs, device=self.device)).cpu().numpy()
        _, returns, *_, info = envs.step(action)

        del info["current_time"]
        del info["n_queries"]

        return {
            key: value
            for key, value in info.items()
            if not key.startswith('_')
        }

    def validate(self) -> dict[str, float]:

        with torch.no_grad():
            val_action = self.policy.greedy(val_obs).cpu().numpy()

        val_envs.reset()
        _, returns, *_, val_info = val_envs.step(val_action)

        del val_info["current_time"]
        del val_info["n_queries"]

        return {
            key: value
            for key, value in val_info.items()
            if not key.startswith('_')
        } | {
            "optimal_gap": (-np.sum(returns) / np.sum(instance_info["optimal"]) - 1).item()
        }

with SchedulingBC(
    states=dataset,
    actions=actions,
    actor=policy,
    actor_optimizer=optimizer,
    device=device,
) as algo:
    algo.learn(
        num_updates      = args.steps,
        global_steps     = args.steps,
        steps_per_update = args.steps_per_update,
        batch_size       = args.batch_size,
        validation_freq  = args.validation_freq,
        log_dir=args.log_dir,
        quiet=args.quiet,
        config=asdict(args)
    )
