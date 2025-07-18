from typing import Callable, Literal, Any
import random as rng


def sample_dirichlet(alpha: list[float]) -> list[float]:
    """
    Samples from a Dirichlet distribution.

    Args:
        alpha (list or tuple): A list or tuple of positive shape parameters
                               (alpha_1, alpha_2, ..., alpha_K).

    Returns:
        list: A list representing a sample from the Dirichlet distribution,
              where the elements sum to 1.
    """
    if not all(a > 0 for a in alpha):
        raise ValueError("All alpha parameters must be positive.")

    k_gamma = [rng.gammavariate(a, 1) for a in alpha]
    sum_k_gamma = sum(k_gamma)

    return [x / sum_k_gamma for x in k_gamma]


def generate_poisson_process(
    loc: int,
    scale: float,
    size: int,
    seed: int | None = None,
) -> list[int]:
    rng.seed(seed)

    process = [loc] * size

    for i in range(1, size):
        process[i] = process[i - 1] + int(rng.expovariate(1 / scale))

    return process
