from typing import Any

from numpy.typing import NDArray
import numpy as np

def confidence_interval(
    metrics: NDArray[np.floating[Any]],
    thresholds: NDArray[np.floating[Any]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    rng = np.random.default_rng(0)

    _, n_samples = metrics.shape

    choices = rng.choice(metrics, size=(n_bootstrap, n_samples), replace=True, axis=-1)
    choices = np.sort(choices, axis=-1)

    ccdfs = 1- ((np.expand_dims(choices, axis=-2) < np.expand_dims(thresholds, axis=-1)).sum(axis=-1) / n_samples)

    alpha = 1 - confidence

    lower = np.quantile(ccdfs, alpha/2, axis=-2)
    upper = np.quantile(ccdfs, 1 - alpha/2, axis=-2)
    mean_ccdf = np.mean(ccdfs, axis=-2)

    return mean_ccdf, lower, upper

