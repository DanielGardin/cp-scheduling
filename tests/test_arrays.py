import pytest
from common import env_setup

def test_numpy_scalar() -> None:
    np = pytest.importorskip("numpy")
    env = env_setup("ta01")

    env.reset()
    obs, reward, terminated, truncated, info = env.step(("execute", np.array(0)))

def test_torch_scalar() -> None:
    torch = pytest.importorskip("torch")

    env = env_setup("ta01")

    env.reset()
    obs, reward, terminated, truncated, info = env.step(("execute", torch.tensor(0)))
