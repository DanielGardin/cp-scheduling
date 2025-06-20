from typing import Any, Optional, MutableSequence, TypeAlias, Callable, Self
from torch.types import Device

from pathlib import Path

import numpy as np

from datetime import datetime

import torch
from torch.nn import Module
from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from abc import ABC, abstractmethod

import logging
import tqdm

from .buffer import Buffer
from .utils import set_seed


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

# This is used to get the number of batches in the minibatch update
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def summary_logs(logs: dict[str, Any], tag: str | None = None) -> dict[str, str]:
    summary: dict[str, str] = {}

    for key, value in logs.items():
        if tag: key = f"{tag}/{key}"

        if isinstance(value, float):
            summary[key] = f"{value:.4f}"

        elif isinstance(value, MutableSequence):
            summary[key] = f"{np.mean(value):.4f} ± {np.std(value):.4f}"

    return summary


def extend_logs(logs: dict[str, list[Any]], new_logs: dict[str, Any]) -> None:
    for key, value in new_logs.items():
        if key not in logs:
            logs[key] = []

        if isinstance(value, float):
            logs[key].append(value)

        elif isinstance(value, MutableSequence):
            logs[key].extend(value)


class BaseAlgorithm(Module, ABC):
    writer: SummaryWriter

    def __init__(self, buffer: Buffer) -> None:
        super().__init__()

        self.buffer = buffer
        self.global_step = 0

    @property
    def device(self) -> Device:
        return next(self.parameters()).device # type: ignore

    def _write_logs(self, logs: dict[str, Any], tag: Optional[str] = None) -> None:
        if not hasattr(self, "writer"):
            return

        for key, value in logs.items():
            if tag: key = f"{tag}/{key}"
            
            if isinstance(value, float):
                self.writer.add_scalar(key, value, self.global_step) # type: ignore

            elif isinstance(value, MutableSequence):
                self.writer.add_scalar(key + "/mean", np.mean(value), self.global_step) # type: ignore
                self.writer.add_scalar(key + "/std", np.std(value), self.global_step) # type: ignore

    def begin_experiment(
            self,
            project_name: str,
            experiment_name: str,
            log_dir: str | Path,
            use_wandb: bool = False,
            config: Optional[dict[str, Any]] = None
        ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        experiment_name = f"{experiment_name}_{timestamp}"

        if use_wandb:
            try:
                import wandb

            except ImportError:
                logger.error("wandb is not installed, please install it to use this feature.")
                return

            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                dir=log_dir,
                config=config,
                sync_tensorboard=True
            )

        self.writer = SummaryWriter(log_dir=log_dir)

    def learn(
            self,
            num_updates: int,
            steps_per_update: int,
            batch_size: int,
            validation_freq: Optional[int] = 1,
            lr_scheduler: Optional[LRScheduler] = None,
            seed: Optional[int] = None
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
        for update in range(1, num_updates+1):
            self.train()

            start_logs = self.on_epoch_start()
            self._write_logs(start_logs, tag="start")

            update_logs.clear()

            n_steps = steps_per_update * ceildiv(len(self.buffer), batch_size)
            with tqdm.tqdm(
                total=n_steps,
                unit=" steps",
                dynamic_ncols=True,
                # ncols=300,
                leave=validation_freq is not None and update % validation_freq == 0
            ) as pbar:
                pbar.set_description(f"Epoch {update}/{num_updates}")

                for _ in range(steps_per_update):
                    for batch in self.buffer.loader(batch_size, self.device):
                        train_logs = self.update(batch)

                        pbar.set_postfix(
                            summary_logs(train_logs)
                        )
                        pbar.update()

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

                    pbar.set_postfix({
                        **summary_logs(start_logs, tag="start"),
                        **summary_logs(update_logs, tag="update"),
                        **summary_logs(end_logs, tag="end"),
                        **summary_logs(val_logs, tag="val")
                    })

            if lr_scheduler is not None:
                lr_scheduler.step()

    def on_session_start(
            self,
            num_updates: int,
            steps_per_update: int,
            batch_size: int
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

    def end_experiment(self) -> None:
        if hasattr(self, "writer"):
            self.writer.close()

    def __enter__(self) -> Self:
        self.train()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.eval()
        self.end_experiment()

