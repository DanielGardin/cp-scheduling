from warnings import warn
from pathlib import Path
import json

from typing import Any, Self
from torch.types import Device, Tensor

import numpy as np

import torch
from torch.nn import Module
from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler

from abc import ABC, abstractmethod

import logging
import tqdm

from .buffer import Buffer
from .utils import get_device
from .protocols import Policy
from .logger import Logger

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)


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


class BaseAlgorithm(Module, ABC):
    config: dict[str, Any]

    def __init__(self, buffer: Buffer, policy: Policy[Tensor, Tensor], device: Device):
        super().__init__()

        self.policy = policy

        self.buffer = buffer
        self.buffer.to(get_device(device))

        self.global_step = 0

        self.config = {}

    @property
    def device(self) -> Device:
        return next(self.parameters()).device

    def learn(
        self,
        num_updates: int | None = None,
        global_steps: int | None = None,
        steps_per_update: int = 4,
        batch_size: int = 1,
        validation_freq: int | None = 1,
        lr_scheduler: LRScheduler | None = None,
        *,
        project_name: str | None = None,
        experiment_name: str = "",
        use_wandb: bool = False,
        log_dir: str | Path | None = None,
        save_model: bool = True,
        quiet: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        logger = Logger(
            project_name=project_name,
            experiment_name=experiment_name,
            log_dir=log_dir,
            use_wandb=use_wandb,
            config=config,
        )

        if global_steps is None:
            if num_updates is None:
                raise ValueError("Either num_updates or global_steps must be provided.")

            global_steps = (
                num_updates
                * steps_per_update
                * ceildiv(self.buffer.capacity, batch_size)
            )

        expected_updates = (
            ceildiv(
                global_steps,
                steps_per_update * ceildiv(self.buffer.capacity, batch_size),
            )
            if num_updates is None
            else num_updates
        )

        self.on_session_start(steps_per_update, batch_size)

        update_count = 0
        while self.global_step < global_steps or (
            num_updates is not None and update_count < num_updates
        ):
            update_count += 1

            start_logs = self.on_epoch_start()
            logger.add_logs(start_logs, step=self.global_step, tag="start")

            n_steps = steps_per_update * ceildiv(len(self.buffer), batch_size)
            with tqdm.tqdm(
                total=n_steps,
                unit=" steps",
                dynamic_ncols=True,
                leave=validation_freq is not None
                and update_count % validation_freq == 0,
                disable=quiet,
            ) as pbar:
                pbar.set_description(f"Epoch {update_count}/{expected_updates}")
                logger.reset_log_accumulator()

                self.train()

                for _ in range(steps_per_update):
                    for batch in self.buffer.loader(batch_size, self.device):
                        train_logs = self.update(batch)
                        logger.accumulate(train_logs)

                        self.global_step += 1
                        pbar.update()

                accumulated_logs = logger.get_accumulator_mean()
                logger.add_logs(accumulated_logs, step=self.global_step, tag="update")

                end_logs = self.on_epoch_end()
                logger.add_logs(end_logs, step=self.global_step, tag="end")

                val_logs = {}
                if validation_freq is not None and update_count % validation_freq == 0:
                    with torch.no_grad():
                        self.eval()

                        val_logs = self.validate()
                        logger.add_logs(val_logs, step=self.global_step, tag="val")

                    pbar.set_postfix(
                        {
                            **summary_logs(start_logs, tag="start"),
                            **logger.get_accumulator_summary(),
                            **summary_logs(end_logs, tag="end"),
                            **summary_logs(val_logs, tag="val"),
                        }
                    )

            if lr_scheduler is not None:
                lr_scheduler.step()

            if num_updates is None:
                expected_updates = ceildiv(
                    global_steps * update_count, self.global_step
                )

        self.on_session_end()
        logger.close()

        if save_model:
            self.eval()
            self.to(device="cpu")

            save_path = Path(logger.get_logdir())

            torch.save(self.policy, save_path / "policy.pkl")

            if isinstance(self.policy, Module):
                torch.save(
                    self.policy.state_dict(),
                    f"{save_path / 'weights.pth'}",
                )

        if config:
            config_path = Path(logger.get_logdir()) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

        self.to(device=self.device)

    def on_session_start(self, steps_per_update: int, batch_size: int) -> None:
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

    def on_session_end(self) -> None:
        pass

    def __enter__(self) -> Self:
        self.train()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.eval()
        self.on_session_end()
