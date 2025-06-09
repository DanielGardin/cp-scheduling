from typing import Any, TypeVar, Callable
from pandas import DataFrame

from gymnasium import Env, Wrapper

from ..common import ProcessTimeAllowedTypes

_Obs = TypeVar("_Obs")
_Act = TypeVar("_Act")

class RandomGeneratorWrapper(Wrapper[_Obs, _Act, _Obs, _Act]):
    """
        A wrapper that generates a random instance of the scheduling problem every
        time the environment is reset.

        Parameters
        ----------
        env : Env
            The scheduling environment to wrap.

        generator : Callable[[int | None], tuple[DataFrame, ProcessTimeAllowedTypes]]
            A callable that generates a random instance and processing times.
            It should accept an optional seed as input.
    """

    def __init__(
        self,
        env: Env[_Obs, _Act],
        generator: Callable[[int | None], tuple[DataFrame, ProcessTimeAllowedTypes]],
    ):
        super().__init__(env)
        self.generator = generator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[_Obs, dict[str, Any]]:
        instance, process_times = self.generator(seed)

        options = {
            "instance": instance,
            "processing_times": process_times,
        }

        return super().reset(seed=seed, options=options)