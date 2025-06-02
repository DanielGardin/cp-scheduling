from __future__ import annotations

from typing import Any, Optional, Mapping, Self
from torch.types import Device

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from datetime import datetime

from tensordict import TensorDict
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter # type: ignore

from abc import ABC, abstractmethod
import logging
import tqdm

from .buffer import Buffer


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

# This is used to get the number of batches in the minibatch update
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


class DummyWriter(SummaryWriter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class Logs:
    def __init__(self, initial_logs: Logs | dict[str, list[float]] | None = None, initial_tag: Optional[str] = None) -> None:
        self.logs: dict[str, list[float]] = {}

        if initial_logs is not None:
            self.extend(initial_logs, initial_tag)


    def log(
            self,
            log_info: Mapping[str, float] | Logs,
            tag: Optional[str] = None
        ) -> Self:
        if isinstance(log_info, Logs):
            self.extend(log_info, tag)


        else:
            for key, value in log_info.items():
                if tag: key = f"{tag}/{key}"

                if key not in self.logs:
                    self.logs[key] = []

                self.logs[key].append(value)

        return self



    def extend(
            self,
            logs: Logs | dict[str, list[float]],
            tag: Optional[str] = None
        ) -> Self:
        if isinstance(logs, Logs):
            logs = logs.logs

        for key, values in logs.items():
            if tag: key = f"{tag}/{key}"

            if key not in self.logs:
                self.logs[key] = []

            self.logs[key].extend(values)

        return self


    def mean_log(self) -> dict[str, Any]:
        return {
            key: np.nanmean(values) for key, values in self.logs.items()
        }


    def std_log(self) -> dict[str, Any]:
        return {
            key: np.nanstd(values) for key, values in self.logs.items()
        }


    def clear(self) -> None:
        self.logs.clear()


    def _write(self, writer: SummaryWriter, global_step: int) -> None:
        for key, values in self.logs.items():
            writer.add_scalar(key, np.mean(values), global_step)
    

    def _summary(self) -> dict[str, str]:
        return {
            key: f"{np.mean(values):.4f} ± {np.std(values):.4f}" if np.std(values) > 0 else f"{np.mean(values):.4f}" for key, values in self.logs.items()
        }

class BaseAlgorithm(nn.Module, ABC):
    def __init__(self, buffer: Buffer) -> None:
        super().__init__()

        self.writer = DummyWriter()
        self.global_step = 0

        self.buffer = buffer


    @property
    def device(self) -> Device:
        return next(self.parameters()).device


    def on_epoch_start(self) -> dict[str, Any] | Logs:
        return {}


    @abstractmethod
    def update(self, batch: TensorDict) -> dict[str, Any]:
        pass


    def on_epoch_end(self) -> dict[str, Any] | Logs:
        return {}
    

    def validate(self) -> dict[str, Any] | Logs:
        return {}


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
            import wandb
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                dir=log_dir,
                config=config,
                sync_tensorboard=True
            )

        self.writer = SummaryWriter(log_dir=log_dir)


    def _write_logs(self, logs: dict[str, Any] | Logs) -> None:
        if isinstance(logs, Logs):
            for key, value in logs.logs.items():
                self.writer.add_scalar(f"{key}/mean", np.mean(value), self.global_step)
                std = np.std(value)

                if std > 0:
                    self.writer.add_scalar(f"{key}/std", std, self.global_step)

        else:
            for key, value in logs.items():
                self.writer.add_scalar(key, value, self.global_step)


    def on_session_start(
            self,
            num_updates: int,
            steps_per_update: int,
            batch_size: int
        ) -> None:
        pass

    def learn(
            self,
            num_updates: int,
            steps_per_update: int,
            batch_size: int,
            validation_freq: Optional[int] = 1,
            lr_scheduler: Optional[LRScheduler] = None,
        ) -> None:
        if isinstance(self.writer, DummyWriter):
            logger.info("Calling learn before begin_experiment, no logging will be done and no weights will be saved.")

        self.on_session_start(num_updates, steps_per_update, batch_size)

        logs = Logs()
        for update in range(1, num_updates+1):
            self.train()
            logs.clear()

            start_logs = self.on_epoch_start()
            logs.log(start_logs, tag="start")
            self._write_logs(start_logs)

            n_steps = steps_per_update * ceildiv(len(self.buffer), batch_size)
            with tqdm.tqdm(total=n_steps, unit=" steps", dynamic_ncols=True, ncols=300) as pbar:
                pbar.set_description(f"Epoch {update}/{num_updates}")

                for _ in range(steps_per_update):
                    for batch in self.buffer.loader(batch_size, self.device):
                        train_logs = self.update(batch)

                        logs.log(train_logs, tag="update")
                        pbar.set_postfix(train_logs)
                        pbar.update()

                        self.global_step += 1

                    # Only the last step should be written
                    self._write_logs(logs)

                end_logs = self.on_epoch_end()
                logs.log(end_logs, tag="end")
                self._write_logs(end_logs)

                if validation_freq is not None and update % validation_freq == 0:
                    val_logs = self.validate()
                    logs.log(val_logs, tag="validation")
                    self._write_logs(val_logs)

                pbar.set_postfix(
                    logs._summary()
                )

            if lr_scheduler is not None:
                lr_scheduler.step()


