from typing import Any, Mapping, Optional, Iterable, Never, Union, Literal
from torch.types import Device, Tensor

import torch
from tensordict import TensorDict

from warnings import warn

IndexType = Union[None, int, slice, Tensor, list[Any], tuple[Any, ...]]


class Buffer:
    def __init__(
            self,
            buffer_size: int,
            buffer_shapes: Mapping[str, tuple[int, ...]],
            device: Device = 'cpu',
            allow_grad: bool = False
        ) -> None:
        self.capacity = buffer_size
        self.device = device

        self.buffer = TensorDict({
                key: torch.empty((buffer_size, *shape), dtype=torch.float32, device=device)
                for key, shape in buffer_shapes.items()
            },
            batch_size=buffer_size,
            device=device
        )

        self.norm: dict[str, tuple[Tensor, Tensor]] = {}

        self.buffer_shapes = buffer_shapes
        self.allow_grad    = allow_grad

        self.initialized = False
        self.clear()


    def __len__(self) -> int:
        return self.current_size


    def to(self, device: Device) -> None:
        if device != self.device:
            self.device = device
            self.buffer = self.buffer.to(device)


    def check_input(self, **kwargs: Tensor) -> Never | None:
        expected_keys = set(self.buffer_shapes.keys())
        given_keys    = set(kwargs.keys())

        if expected_keys == given_keys:
            return None

        error_message = "Got a mismatch on buffer keys, "

        lacking_keys = expected_keys - given_keys
        excess_keys  = given_keys - expected_keys

        if lacking_keys:
            error_message += f"missing keys {', '.join(lacking_keys)},"

        if excess_keys:
            error_message += f"unexpected keys: {', '.join(excess_keys)}."

        raise TypeError(error_message)


    def initialize(self, **kwargs: Tensor) -> None:
        for key in self.buffer_shapes.keys():
            self.buffer[key] = self.buffer[key].to(kwargs[key].dtype)

        self.initialized = True


    def clear(self) -> None:
        self.idx = 0
        self.current_size = 0

        self.norm.clear()


    def add(self, **kwargs: Tensor) -> None:
        self.check_input(**kwargs)

        # Correct tensor shapes
        for key, value in kwargs.items():
            kwargs[key] = value.reshape(-1, *self.buffer_shapes[key])

        first_dims = [x.shape[0] for x in kwargs.values()]
        n_samples  = first_dims[0]

        if not all(x == n_samples for x in first_dims):
            raise ValueError("All tensors must have the same batch size")

        if not self.initialized:
            self.initialize(**kwargs)

        idxs = torch.arange(self.idx, self.idx + n_samples) % self.capacity

        for key, value in kwargs.items():
            if value.requires_grad and not self.allow_grad:
                warn("Adding a tensor with grad to the buffer, but allow_grad is set to False. "
                     "The tensor will be detached before being added to the buffer.")
                value = value.detach()

            self.buffer[key][idxs] = value.to(self.device)

        self.idx = (self.idx + n_samples) % self.capacity
        self.current_size = min(self.current_size + n_samples, self.capacity)


    def normalize(
            self,
            keys: Optional[Iterable[str]] = None,
            kind: Literal['minmax', 'standard'] = 'minmax',
            eps: float = 1e-6
        ) -> None:

        if keys is None:
            keys = self.buffer_shapes.keys()
        
        elif isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if key not in self.buffer_shapes:
                raise ValueError(f"Key {key} not found in buffer shapes")
        
            if kind == 'minmax':
                min_val, _ = self[key].min(dim=0)
                max_val, _ = self[key].max(dim=0)

                shift = min_val
                scale = max_val - shift

            elif kind == 'standard':
                shift = self[key].mean(dim=0)
                scale = self[key].std(dim=0)

            # If the scale is zero, then there's no variation
            scale[scale == 0] = eps

            self.norm[key] = (shift, scale)


    def _apply_norm(self, buffer: TensorDict) -> TensorDict:
        for key, (shift, scale) in self.norm.items():
            buffer[key] = (buffer[key] - shift) / scale

        return buffer


    def sample(self, batch_size: int, device: Optional[Device] = None) -> TensorDict:
        idxs = torch.randint(0, self.current_size, (batch_size,))

        buffer: TensorDict = self.buffer[idxs]

        if device is not None and device != self.device:
            buffer = buffer.to(device)

        return self._apply_norm(buffer)


    def loader(self, batch_size: int, device: Optional[Device] = None) -> Iterable[TensorDict]:
        idxs = torch.randperm(self.current_size)
        batch: TensorDict

        i = 0
        for i in range(0, self.current_size - batch_size + 1, batch_size):
            batch = self.buffer[idxs[i:i+batch_size]]

            if device is not None and device != self.device:
                batch = batch.to(device)

            yield self._apply_norm(batch)

        # Yield the (potentially incomplete) last batch
        batch = self.buffer[idxs[i:self.current_size]]

        if device is not None and device != self.device:
            batch = batch.to(device)

        yield self._apply_norm(batch)


    def __getitem__(self, idx: str) -> Tensor:
        return self.buffer[idx][:self.current_size] # type: ignore
