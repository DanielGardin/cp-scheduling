from typing import Any, Literal
from collections.abc import Callable

from functools import partial

from numpy.typing import NDArray, ArrayLike
import numpy as np


def ccdf(
    metrics: NDArray[np.floating[Any]],
    thresholds: ArrayLike | None = None,
    axis: int = -1,
) -> NDArray[np.floating[Any]]:
    """
    Compute the complementary cumulative distribution function (CCDF) of the given metrics.
    The CCDF is defined as the fraction of instances for which the metric is greater than or
    equal to a given threshold.

    This metric is very useful for evaluating policy performance due to the stochastic nature of
    policies. The CCDF allows us to see how many instances have as good performance as a given threshold,
    for example, Human Performance in Atari, or the optimal performance in a given instance.

    Parameters
    ----------
    metrics : NDArray[np.floating[Any]]
        The metrics to compute the CCDF for. The shape of the array should be (*batch, n),
        where *batch represents any number of leading dimensions and n is the number of instances.

    thresholds : ArrayLike | None, optional
        The thresholds to compute the CCDF for. If None, the thresholds will be computed as
        a linear space between the minimum and maximum values of the metrics. If provided, it should
        be an array-like object of shape (t,), where t is the number of thresholds.

    axis : int, optional
        The axis along which to compute the CCDF. Default is -1, which means the
        last axis of the metrics array.

    Returns
    -------
    NDArray[np.floating[Any]]
        The CCDF values for the given thresholds. The shape of the returned array will be
        (*batch, t), where *batch is the same as in the input metrics and t is the number of thresholds.
    """

    if thresholds is None:
        thresholds = np.linspace(np.min(metrics), np.max(metrics), num=100)

    else:
        thresholds = np.asarray(thresholds)

    metrics = np.moveaxis(metrics, axis, -1)  # shape: (*batch, n)
    metrics = np.sort(metrics, axis=-1)

    n = metrics.shape[-1]

    cmp = (
        metrics[..., np.newaxis] >= thresholds[np.newaxis, ...]
    )  # shape: (*batch, n, t)

    ccdf: NDArray[Any] = np.mean(cmp, axis=-2)  # shape: (*batch, t)

    return ccdf


Statistics = Literal["mean", "median", "min", "max", "ccdf"]
Statistic_FN = Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]


def confidence_interval(
    metrics: NDArray[np.floating[Any]],
    n_bootstrap: int = 1000,
    statistic: Statistics | str | Statistic_FN | None = "mean",
    axis: int = -1,
    confidence: float = 0.95,
    seed: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    """
    Compute the confidence interval for a given statistic of the metrics using bootstrap sampling.

    Parameters
    ----------
    metrics : NDArray[np.floating[Any]]
        The metrics to compute the confidence interval for.

    n_bootstrap : int, optional
        The number of bootstrap samples to use for estimating the confidence interval. Default is 1000

    statistic : Statistics | str | Statistic_FN, optional
        The statistic to compute for each bootstrap sample. Can be one of the predefined statistics
        ("mean", "median", "min", "max", "ccdf") or a custom function that takes an array and returns a statistic.
        if None, returns the spread confidence interval

    axis : int, optional
        The axis along which to compute the statistic. Default is -1, which means the last
        axis of the metrics array.

    confidence : float, optional
        The confidence level for the interval. Default is 0.95, which corresponds to a 95% confidence interval.

    seed : int | None, optional
        Random seed for reproducibility. If None, the random number generator will not be seeded.

    *args : Any
        Additional positional arguments to pass to the statistic function.

    **kwargs : Any
        Additional keyword arguments to pass to the statistic function.

    Returns
    -------
    tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
        A tuple containing three arrays:
        - The computed statistic for each bootstrap sample.
        - The lower bound of the confidence interval.
        - The upper bound of the confidence interval.
    """

    rng = np.random.default_rng(seed)

    if isinstance(statistic, str):
        if hasattr(np, statistic):
            statistic = partial(getattr(np, statistic), axis=-1)

        elif statistic == "ccdf":
            statistic = ccdf

        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    elif statistic is None:
        statistic = lambda x: x

    metrics = np.moveaxis(metrics, axis, -1)  # shape: (*batch, n_samples)
    n_samples = metrics.shape[-1]

    idx = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))
    samples = metrics[..., idx]
    samples = np.moveaxis(samples, -2, 0)  # shape: (n_bootstrap, *batch, n_samples)

    statistics = statistic(samples, *args, **kwargs)  # shape: (n_bootstrap, ...)

    alpha = 1 - confidence

    lower = np.quantile(statistics, alpha / 2, axis=0)
    upper = np.quantile(statistics, 1 - alpha / 2, axis=0)

    return statistics, lower, upper
