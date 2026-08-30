"""Cyclic instance generator."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.constants import EzPickle
from cpscheduler.environment.utils import InstanceTypes
from cpscheduler.environment.utils.protocols import InstanceGenerator


class CyclicGenerator(EzPickle, InstanceGenerator):
    """Cyclic instance generator.

    The generator cycles a collection of instances, loading the next instance
    in the sequence for each reset call.
    """

    _instance_seq: list[InstanceTypes]
    _idx: int

    def __init__(self, instances: Iterable[InstanceTypes]) -> None:
        self._instance_seq = list(instances)
        self._idx = -1

    @override
    def sample(
        self,
        seed: int | None = None,
    ) -> InstanceTypes:
        self._idx = (self._idx + 1) % len(self._instance_seq)
        return self._instance_seq[self._idx]
