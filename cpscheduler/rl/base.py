from warnings import warn
from pathlib import Path

from typing import Any, Self
from collections.abc import Iterable
from torch.types import Device, Tensor

import numpy as np

from datetime import datetime

import torch
from torch.nn import Module
from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from abc import ABC, abstractmethod

import logging
import tqdm

from .buffer import Buffer
from .utils import set_seed, get_device
from .protocols import Policy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# This is used to get the number of batches in the minibatch update
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def summary_logs(logs: dict[str, Any], tag: str | None = None) -> dict[str, str]:
    summary: dict[str, str] = {}

    for key, value in logs.items():
        if tag:
            key = f"{tag}/{key}"

        try:
            mean = np.mean(value)
            std = np.std(value)

            summary[key] = f"{mean:.4f}" + ("" if std <= 1e-8 else f" ± {std:.4f}")

        except TypeError:
            warn(f"Could not summarize log {key} with type {type(value)}.")
            # If value is not a sequence, we handle it separately
            pass

    return summary


def extend_logs(logs: dict[str, list[Any]], new_logs: dict[str, Any]) -> None:
    for key, value in new_logs.items():
        if key not in logs:
            logs[key] = []

        if isinstance(value, float):
            logs[key].append(value)

        elif isinstance(value, Iterable):
            logs[key].extend(value)


class BaseAlgorithm(Module, ABC):
    writer: SummaryWriter

    def __init__(
        self, buffer: Buffer, policy: Policy[Tensor, Tensor], device: Device
    ) -> None:
        super().__init__()

        self.policy = policy

        self.buffer = buffer
        self.buffer.to(get_device(device))

        self.global_step = 0

        self.metric: str = ""
        self.minimize: bool = True
        self.best_running_metric: float = float("inf")
        self.checkpoint_path: Path | None = None

    @property
    def device(self) -> Device:
        return next(self.parameters()).device

    def _write_logs(self, logs: dict[str, Any], tag: str | None = None) -> None:
        if not hasattr(self, "writer"):
            return

        for key, value in logs.items():
            if tag:
                key = f"{tag}/{key}"

            try:
                mean = np.mean(value)

                self.writer.add_scalar(key, mean, global_step=self.global_step)

            except TypeError:
                warn(f"Could not summarize log {key} with type {type(value)}.")
                # If value is not a sequence, we handle it separately
                pass

    def begin_experiment(
        self,
        project_name: str,
        experiment_name: str = "experiment",
        log_dir: str | Path | None = None,
        use_wandb: bool = False,
        config: dict[str, Any] | None = None,
        save_model: bool = True,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        experiment_name = f"{experiment_name}_{timestamp}"
        if log_dir is not None:
            log_dir = Path(log_dir) / experiment_name

        if use_wandb:
            try:
                import wandb

            except ImportError:
                logger.error(
                    "wandb is not installed, please install it to use this feature."
                )
                return

            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                dir=log_dir,
                config=config,
                sync_tensorboard=True,
            )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_model = save_model
        self.config = config

    def set_objective_metric(
        self,
        objective_metric: str,
        minimize: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.metric = objective_metric
        self.minimize = minimize

        if checkpoint_path is None:
            if hasattr(self, "writer"):
                self.checkpoint_path = Path(self.writer.get_logdir()) / "checkpoint.pth"

        else:
            self.checkpoint_path = Path(checkpoint_path)

    def learn(
        self,
        num_updates: int,
        steps_per_update: int,
        batch_size: int,
        validation_freq: int | None = 1,
        lr_scheduler: LRScheduler | None = None,
        seed: int | None = None,
    ) -> None:
        if not hasattr(self, "writer"):
            logger.info(
                "Calling learn before begin_experiment, no logging will be done"
                "and no weights will be saved."
            )

            self.writer = SummaryWriter()

        if seed is not None:
            set_seed(seed)

        self.on_session_start(num_updates, steps_per_update, batch_size)

        update_logs: dict[str, list[Any]] = {}
        for update in range(1, num_updates + 1):
            self.train()

            start_logs = self.on_epoch_start()
            self._write_logs(start_logs, tag="start")

            update_logs.clear()

            n_steps = steps_per_update * ceildiv(len(self.buffer), batch_size)
            with tqdm.tqdm(
                total=n_steps,
                unit=" steps",
                dynamic_ncols=True,
                leave=validation_freq is not None and update % validation_freq == 0,
            ) as pbar:
                pbar.set_description(f"Epoch {update}/{num_updates}")

                for _ in range(steps_per_update):
                    for batch in self.buffer.loader(batch_size, self.device):
                        train_logs = self.update(batch)

                        pbar.update()
                        pbar.set_postfix(summary_logs(train_logs))

                        extend_logs(update_logs, train_logs)

                        self.global_step += 1

                self._write_logs(update_logs, tag="update")
                end_logs = self.on_epoch_end()
                self._write_logs(end_logs, tag="end")

                val_logs = {}
                if validation_freq is not None and update % validation_freq == 0:
                    with torch.no_grad():
                        self.eval()
                        val_logs = self.validate()
                        self._write_logs(val_logs, tag="val")

                    pbar.set_postfix(
                        {
                            **summary_logs(start_logs, tag="start"),
                            **summary_logs(update_logs, tag="update"),
                            **summary_logs(end_logs, tag="end"),
                            **summary_logs(val_logs, tag="val"),
                        }
                    )

            if self.metric:
                log: dict[str, Any] = {}
                for log in [update_logs, end_logs, val_logs]:
                    if self.metric not in log:
                        continue

                    metric_value = float(np.mean(log[self.metric]))
                    if self.minimize:
                        if metric_value < self.best_running_metric:
                            self.best_running_metric = metric_value
                            if self.checkpoint_path is not None:
                                self.save_checkpoint(self.checkpoint_path)

            if lr_scheduler is not None:
                lr_scheduler.step()

    def on_session_start(
        self, num_updates: int, steps_per_update: int, batch_size: int
    ) -> None:
        pass

    def on_epoch_start(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def update(self, batch: TensorDict) -> dict[str, Any]:
        pass

    def on_epoch_end(self) -> dict[str, Any]:
        return {}

    def validate(self) -> dict[str, Any]:
        return {}

    def save_checkpoint(self, filename: str | Path) -> None:
        if isinstance(self.policy, Module):
            torch.save(
                self.policy.state_dict(),
                f"{filename}.pth",
            )

        else:
            torch.save(self.policy, f"{filename}.pkl")

        logger.info(f"Checkpoint saved to {filename}")

    def end_experiment(self) -> None:
        if self.save_model:
            model_path = Path(self.writer.get_logdir()) / "policy"
            self.save_checkpoint(model_path)

        if self.config is not None:
            config_path = Path(self.writer.get_logdir()) / "config.json"
            with open(config_path, "w") as f:
                import json

                json.dump(self.config, f, indent=4)

        if hasattr(self, "writer"):
            self.writer.close()

    def __enter__(self) -> Self:
        self.train()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.eval()
        self.end_experiment()

    def get_logs(self) -> dict[str, dict[int, Any]]:
        if not hasattr(self, "writer"):
            logger.warning("No writer found, returning empty logs.")
            return {}

        event_acc = EventAccumulator(str(self.writer.get_logdir()))
        event_acc.Reload()

        tags = event_acc.Tags()["scalars"]

        return {
            tag: {event.step: event.value for event in event_acc.Scalars(tag)}
            for tag in tags
        }

    def get_last_metrics(self) -> dict[str, Any]:
        logs = self.get_logs()
        if not logs:
            return {}

        return {tag: values[max(values.keys())] for tag, values in logs.items()}
