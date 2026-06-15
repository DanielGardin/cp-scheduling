"""Continuous distributions commonly used in scheduling instance generation."""

from random import Random
from typing import override

from cpscheduler.instances.distributions.base import Distribution, Sampler


class Uniform(Distribution[float]):
    """Continuous uniform distribution on [low, high]."""

    low: float
    high: float

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        """Initialize a Uniform distribution.

        Parameters
        ----------
        low : float
            Lower bound of the distribution (inclusive).

        high : float
            Upper bound of the distribution (inclusive).

        Raises
        ------
        ValueError
            If low > high.

        """
        if low > high:
            raise ValueError(
                f"Expected low <= high, received {low} > {high}."
            )

        self.low = float(low)
        self.high = float(high)

    @override
    def sample(self, rng: Random, **context: object) -> float:
        return rng.uniform(self.low, self.high)

    @override
    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


class Normal(Distribution[float]):
    """Normal distribution."""

    mean: float
    stdev: float

    def __init__(self, mean: float, stdev: float) -> None:
        """Initialize a Normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the distribution.

        stdev : float
            Standard deviation of the distribution. Must be > 0.

        Raises
        ------
        ValueError
            If `stdev` is not > 0.

        """
        if stdev <= 0:
            raise ValueError(
                f"Expected stdev > 0, received {stdev}."
            )

        self.mean = float(mean)
        self.stdev = float(stdev)

    @override
    def sample(self, rng: Random, **context: object) -> float:
        return rng.gauss(self.mean, self.stdev)

    @override
    def __repr__(self) -> str:
        return (
            f"Normal("
            f"mean={self.mean}, "
            f"stdev={self.stdev})"
        )

class Exponential(Distribution[float]):
    """
    Exponential distribution.

    Parameterized by its mean (scale), not its rate.
    """

    scale: float

    def __init__(self, scale: float) -> None:
        """Initialize an Exponential distribution.

        Parameters
        ----------
        scale : float
            Mean of the distribution. Must be > 0.

        Raises
        ------
        ValueError
            If `scale` is not > 0.
        """
        if scale <= 0:
            raise ValueError(
                f"Expected scale > 0, received {scale}."
            )

        self.scale = float(scale)

    @override
    def sample(self, rng: Random, **context: object) -> float:
        return rng.expovariate(1.0 / self.scale)

    @override
    def __repr__(self) -> str:
        return f"Exponential({self.scale})"

class Beta(Distribution[float]):
    """Beta distribution scaled to [low, high]."""

    alpha: float
    beta: float
    low: float
    high: float

    def __init__(
        self,
        alpha: float,
        beta: float,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        """Initialize a Beta distribution.

        Parameters
        ----------
        alpha : float
            Alpha (shape) parameter. Must be > 0.

        beta : float
            Beta (shape) parameter. Must be > 0.

        low : float
            Lower bound of the distribution (inclusive).

        high : float
            Upper bound of the distribution (inclusive).

        Raises
        ------
        ValueError
            If `alpha` or `beta` is not > 0, or if `low` >= `high`.

        """
        if alpha <= 0:
            raise ValueError(
                f"Expected alpha > 0, received {alpha}."
            )

        if beta <= 0:
            raise ValueError(
                f"Expected beta > 0, received {beta}."
            )

        if low >= high:
            raise ValueError(
                f"Expected low < high, received {low} >= {high}."
            )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)

    @override
    def sample(self, rng: Random, **context: object) -> float:
        value = rng.betavariate(
            self.alpha,
            self.beta,
        )

        return self.low + value * (
            self.high - self.low
        )

    @override
    def __repr__(self) -> str:
        return (
            f"Beta("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"low={self.low}, "
            f"high={self.high})"
        )


class Dirichlet(Sampler[list[float]]):
    """Dirichlet distribution."""

    alpha: list[float]

    def __init__(self, alpha: list[float]) -> None:
        """Initialize a Dirichlet distribution.

        Parameters
        ----------
        alpha : list[float]
            List of alpha (shape) parameters. Each must be > 0.

        Raises
        ------
        ValueError
            If any `alpha` parameter is not > 0.

        """
        if not all(a > 0 for a in alpha):
            raise ValueError(
                f"Expected all alpha parameters > 0, received {alpha}."
            )

        self.alpha = [float(a) for a in alpha]

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (len(self.alpha),)

    @override
    def sample(self, rng: Random, **context: object) -> list[float]:
        k_gamma = [rng.gammavariate(a, 1) for a in self.alpha]
        sum_k_gamma = sum(k_gamma)

        return [x / sum_k_gamma for x in k_gamma]

