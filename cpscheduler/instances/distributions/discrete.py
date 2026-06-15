"""Discrete distributions commonly used in scheduling instance generation."""

import math
from bisect import bisect
from collections.abc import Iterable
from random import Random
from typing import TypeVar, override

from cpscheduler.instances.distributions.base import Distribution, Sampler


class UniformInt(Distribution[int]):
    """Discrete uniform distribution on [low, high]."""

    low: int
    high: int

    def __init__(self, low: int, high: int) -> None:
        """Initialize a UniformInt distribution.

        Parameters
        ----------
        low : int
            Lower bound of the distribution (inclusive).

        high : int
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

        self.low = low
        self.high = high

    @override
    def sample(self, rng: Random, **context: object) -> int:
        return rng.randint(self.low, self.high)

    @override
    def __repr__(self) -> str:
        return f"UniformInt({self.low}, {self.high})"


class Bernoulli(Distribution[bool]):
    """Bernoulli distribution."""

    p: float

    def __init__(self, p: float) -> None:
        """Initialize a Bernoulli distribution.

        Parameters
        ----------
        p : float
            Probability of success (True).

        Raises
        ------
        ValueError
            If p is not in [0, 1].

        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Expected p in [0, 1], received {p}."
            )

        self.p = p

    @override
    def sample(self, rng: Random, **context: object) -> bool:
        return rng.random() < self.p

    @override
    def __repr__(self) -> str:
        return f"Bernoulli({self.p})"

_T = TypeVar("_T")

class Choice(Sampler[_T]):
    """Uniform sampling from a finite collection."""

    values: tuple[_T, ...]
    weights: tuple[float, ...]

    def __init__(
        self,
        values: Iterable[_T],
        weights: Iterable[float] | None = None
    ) -> None:
        """Initialize a Choice distribution.

        Parameters
        ----------
        values : Iterable[_T]
            The finite collection of values to sample from.

        weights : Iterable[float] | None, optional
            Optional weights for the values. If `None`, samples uniformly.

        Raises
        ------
        ValueError
            If `weights` is not `None` and contains negative values, or if
            `values` is empty.

        """
        self.values = tuple(values)
        n_categories = len(self.values)

        if weights is None:
            weights = [1] * n_categories

        elif any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative.")

        self.weights = tuple(weights)

    @override
    def sample(self, rng: Random, **context: object) -> _T:
        return rng.choices(
            self.values,
            weights=self.weights,
            k=1,
        )[0]

    @override
    def __repr__(self) -> str:
        return f"Choice({list(self.values)!r})"

class Categorical(Choice[int]):
    """Uniform categorical distribution."""

    n_categories: int

    def __init__(
        self,
        n_categories: int,
        weights: Iterable[float] | None = None
    ) -> None:
        """Initialize a Categorical distribution.

        Parameters
        ----------
        n_categories : int
            Number of categories (values will be 0 to n_categories-1).

        weights : Iterable[float] | None, optional
            Optional weights for the categories. If `None`, samples uniformly.

        Raises
        ------
        ValueError
            If `n_categories` is not positive, or if `weights` is not `None`
            and contains negative values.

        """
        super().__init__(range(n_categories), weights)
        self.n_categories = n_categories

    @override
    def __repr__(self) -> str:
        return f"Categorical({self.n_categories})"

class Geometric(Distribution[int]):
    """Geometric distribution."""

    p: float

    def __init__(self, p: float) -> None:
        """Initialize a Geometric distribution.

        Parameters
        ----------
        p : float
            Probability of success on each trial.

        Raises
        ------
        ValueError
            If p is not in (0, 1].

        """
        if not 0.0 < p <= 1.0:
            raise ValueError(
                f"Expected p in (0, 1], received {p}."
            )

        self.p = p

    @override
    def sample(self, rng: Random, **context: object) -> int:
        if self.p == 1.0:
            return 0

        return math.floor(
                math.log(rng.random()) /
                math.log(1.0 - self.p)
            )

    @override
    def __repr__(self) -> str:
        return f"Geometric({self.p})"


class Poisson(Distribution[int]):
    """Poisson distribution."""

    rate: float
    _exp_neg_rate: float

    def __init__(self, rate: float) -> None:
        """Initialize a Poisson distribution.

        Parameters
        ----------
        rate : float
            Rate (lambda) of the distribution.

        Raises
        ------
        ValueError
            If `rate` is not > 0.

        """
        if rate <= 0:
            raise ValueError(
                f"Expected rate > 0, received {rate}."
            )

        self.rate = rate
        self._exp_neg_rate = math.exp(-rate)

    @override
    def sample(self, rng: Random, **context: object) -> int:
        k = 0
        p = 1.0

        while p > self._exp_neg_rate:
            k += 1
            p *= rng.random()

        return k - 1

    @override
    def __repr__(self) -> str:
        return f"Poisson({self.rate})"


class Multinomial(Sampler[list[int]]):
    """Multinomial distribution."""

    n: int
    pvals: tuple[float, ...]

    _cdf: tuple[float, ...]

    def __init__(self, n: int, pvals: Iterable[float]) -> None:
        """Initialize a Multinomial distribution.

        Parameters
        ----------
        n : int
            Number of trials.

        pvals : Iterable[float]
            Probabilities of each category (need not sum to 1).

        Raises
        ------
        ValueError
            If `n` is negative, or if any probability in `pvals` is negative.

        """
        if n < 0:
            raise ValueError(f"Expected n >= 0, received {n}.")

        self.n = n
        self.pvals = tuple(pvals)

        cdf = [0.0] * len(self.pvals)
        cumulative = 0.0
        for i, p in enumerate(self.pvals):
            if p < 0:
                raise ValueError(
                    f"Expected non-negative probabilities, received {self.pvals}."
                )

            cumulative += p
            cdf[i] = cumulative

        self._cdf = tuple(cdf)

    @override
    def sample(self, rng: Random, **context: object) -> list[int]:
        counts = [0] * len(self.pvals)

        cdf = self._cdf
        total = cdf[-1]

        for _ in range(self.n):
            u = rng.random() * total
            category = bisect(cdf, u)
            counts[category] += 1

        return counts

    @override
    def __repr__(self) -> str:
        return f"Multinomial(n={self.n}, pvals={list(self.pvals)!r})"
