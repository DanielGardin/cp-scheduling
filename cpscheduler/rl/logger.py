from warnings import warn

from pathlib import Path
from typing import Any, SupportsFloat, SupportsInt
from typing_extensions import TypeAlias

from uuid import uuid4

from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch.utils.tensorboard.summary import hparams

import numpy as np
from torch import Tensor

from datetime import datetime
from dataclasses import dataclass

try:
    import wandb

except ImportError:
    wandb = None  # type: ignore[assignment]


@dataclass
class Welford:
    mean: float = 0.0
    count: int = 0
    sqrd_sum: float = 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0

        std: float = (self.sqrd_sum / (self.count - 1)) ** 0.5

        return std

    def update(self, value: SupportsFloat) -> None:
        value = float(value)

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sqrd_sum += delta * (value - self.mean)


class Logger(SummaryWriter):
    def __init__(
        self,
        project_name: str | None = None,
        experiment_name: str = "",
        log_dir: str | Path | None = None,
        use_wandb: bool = False,
        config: dict[str, Any] | None = None,
    ):
        timestamp = datetime.now(tz=datetime.now().astimezone().tzinfo).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        experiment_name = (
            f"{experiment_name}_{timestamp}" if experiment_name else timestamp

        )

        if log_dir is not None:
            base_dir = Path(log_dir)

            log_dir = base_dir / f"{experiment_name}_{uuid4().hex[:4]}"

        self.config = config or {}
        self.use_wandb = use_wandb
        if use_wandb:
            self.initialize_wandb(
                project_name=project_name,
                experiment_name=experiment_name,
                config=config,
            )


        super().__init__(log_dir=log_dir)

        self.accumulator: dict[str, Welford] = {}

    def initialize_wandb(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it to use this feature."
            )

        wandb.init(
            project=project_name,
            name=experiment_name,
            dir=self.get_logdir(),
            config=config,
            sync_tensorboard=True,
        )

    def accumulate(self, logs: dict[str, Any]) -> None:
        for key, value in logs.items():
            if isinstance(value, SupportsFloat):
                if key not in self.accumulator:
                    self.accumulator[key] = Welford()

                self.accumulator[key].update(value)

    def reset_log_accumulator(self) -> None:
        self.accumulator.clear()

    def get_accumulator_mean(self) -> dict[str, float]:
        return {key: welford.mean for key, welford in self.accumulator.items()}

    def get_accumulator_summary(self) -> dict[str, str]:
        return {
            key: f"{w.mean:.4f}" + ("" if w.std <= 1e-8 else f" ± {w.std:.4f}")
            for key, w in self.accumulator.items()
        }

    def add_logs(
        self,
        logs: dict[str, Any],
        step: int | None = None,
        tag: str | None = None,
    ) -> None:
        for key, value in logs.items():
            if tag:
                key = f"{tag}/{key}"

            if isinstance(value, (int, float)):
                self.add_scalar(key, value, global_step=step)

            elif isinstance(value, (list, np.ndarray)):
                self.add_histogram(key, value, global_step=step)
                self.add_scalar(f"{key}/mean", np.mean(value), global_step=step)

            else:
                warn(f"Unsupported log type for key {key}: {type(value)}. Skipping.")

    def close(self) -> None:
        if self.use_wandb and wandb is not None:
            wandb.finish()

        super().close()
