from typing import Protocol, TypeVar
from torch.types import Tensor

ObsT_C = TypeVar("ObsT_C", contravariant=True)
ActT_ = TypeVar("ActT_")


class Policy(Protocol[ObsT_C, ActT_]):
    def get_action(self, x: ObsT_C) -> tuple[ActT_, Tensor]: ...

    def log_prob(self, x: ObsT_C, action: ActT_) -> Tensor: ...

    def greedy(self, x: ObsT_C) -> ActT_: ...


ActT_C = TypeVar("ActT_C", contravariant=True)


class Critic(Protocol[ObsT_C, ActT_C]):
    def get_value(self, x: ObsT_C, action: ActT_C) -> Tensor: ...

    def get_values(self, x: ObsT_C) -> Tensor: ...
