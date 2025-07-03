from typing import Any
from collections.abc import Iterable
from pandas import DataFrame
from numpy.typing import NDArray

import numpy as np
import pandas as pd

# TODO: Remove pandas dependency


def mean_point(arr: NDArray[Any]) -> NDArray[Any]:
    a = np.insert(arr, 0, 0)

    return a[:-1] + np.diff(a) / 2  # type: ignore[no-any-return]


def customer_scheduling_instance(
    base_instances: Iterable[DataFrame],
    n_customers: int,
    sort_noise: float = 0.0,
    *,
    weight_feature: str = "weight",
    customer_feature: str = "customer",
    seed: int | None = None,
) -> tuple[list[DataFrame], DataFrame]:
    """
    Adapt a set of scheduling instances to a customer scheduling instance.
    The customer scheduling instance is a scheduling instance where the jobs are
    owned by customers, who have different weights.

    Given an instance of a α | β | Σ wj h(.) scheduling problem, we create a new customer
    feature such that jobs with equal weights are assigned to customers, stabilishing a
    relation between the customer and the job weights.

    The assumptions are:
    - The customer presence is sampled uniformly from a simplex.
    - Customer weights are correlated to the number of jobs assigned to them (this correlation
      can be controlled by the `sort_noise` parameter).


    Parameters
    ----------
    base_instance : DataFrame
        The base instance to adapt.

    """
    base_instance = pd.concat(
        [
            instance.assign(instance_id=i)
            for i, instance in enumerate(base_instances, 1)
        ],
        ignore_index=True,
    )

    rng = np.random.default_rng(seed)

    weights = np.sort(base_instance[weight_feature].unique())

    weights_distribution = (
        base_instance[weight_feature].value_counts().sort_index().to_numpy()
    )
    weights_distribution = weights_distribution / np.sum(weights_distribution)

    # Customer presences are sampled uniformly from a simplex
    customer_presence = rng.dirichlet(np.ones(n_customers))

    customers_order = np.argsort(
        np.log(customer_presence) + sort_noise * rng.gumbel(0, 1, n_customers)
    )

    cumulative_presence = mean_point(np.cumsum(customer_presence[customers_order]))

    assignment = np.searchsorted(cumulative_presence, np.cumsum(weights_distribution))
    n_customers_per_weight = np.diff(np.insert(assignment, 0, 0))

    # Check if any weight bin has no customers assigned and tweak the boundaries
    for i in range(len(n_customers_per_weight) - 1, 0, -1):
        current = n_customers_per_weight[i]

        if current <= 0:
            required = 1 - current

            n_customers_per_weight[i] += required
            n_customers_per_weight[i - 1] -= required

    customer_information = DataFrame(
        {weight_feature: 0, "n_jobs": 0, "presence": customer_presence}
    )

    idx = 0
    for weight_bin, weight in enumerate(weights):
        n_jobs = len(base_instance[base_instance[weight_feature] == weight])
        n_customers_in_bin = n_customers_per_weight[weight_bin]

        customers = customers_order[idx : idx + n_customers_in_bin]
        bin_presence = customer_presence[customers] / customer_presence[customers].sum()

        jobs_distribution = (
            rng.multinomial(n_jobs - n_customers_in_bin, bin_presence) + 1
        )

        idx += int(n_customers_in_bin)

        customer_information.loc[customers.tolist(), weight_feature] = weight
        customer_information.loc[customers.tolist(), "n_jobs"] = jobs_distribution

        base_instance.loc[base_instance[weight_feature] == weight, customer_feature] = (
            rng.permutation(np.repeat(customers, jobs_distribution))
        )

    weight_norm = customer_information[weight_feature].sum()

    base_instance[weight_feature] = base_instance[weight_feature] / weight_norm

    customer_information[f"{weight_feature}_original"] = customer_information[
        weight_feature
    ]
    customer_information[weight_feature] = (
        customer_information[weight_feature] / weight_norm
    )

    base_instance[customer_feature] = base_instance[customer_feature].astype(int)

    list_instances = [
        instance.drop(columns=["instance_id"])
        for _, instance in base_instance.groupby("instance_id")
    ]

    return list_instances, customer_information
