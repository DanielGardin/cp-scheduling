from typing import Any, Optional, Literal
from numpy.typing import NDArray
from pandas import DataFrame

from pathlib import Path

import numpy as np


def read_instance(path: Path | str) -> tuple[DataFrame, dict[str, Any]]:
    """
    Reads an instance from a file. The file must be in the Taillard format, with the following structure:
    - The first line contains the number of jobs and the number of machines in the instance.
    - The following lines contain the machines and processing times for each operation in the instance. Each line corresponds to a job, and the operations data are separated by a space.

    Parameters
    ----------
    path : str or Path
        Path to the file containing the instance data.
    
    Returns
    -------
    instance : pandas.DataFrame
        DataFrame with the instance data in the following columns:
        - job: Job ID.
        - operation: Operation ID.
        - machine: Machine ID.
        - processing_time: Processing time for the operation.
    
    metadata : dict
        Dictionary with metadata about the instance.
    """

    with open(path, 'r') as f:
        n_jobs, n_machines = map(int, f.readline().split())

        # TODO: Allow metadata to be read from the file

        data: list[list[int]] = []

        for job_id in range(n_jobs):
            job_info = f.readline()
            
            job_data = list(map(int, job_info.split()))

            for operation_id, (machine_id, processing_time) in enumerate(zip(job_data[::2], job_data[1::2])):
                data.append([job_id, operation_id, machine_id, processing_time])


        instance = DataFrame(data, columns=['job', 'operation', 'machine', 'processing_time'])

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    return instance, metadata

def generate_taillard_instance(
        n_jobs: int,
        n_machines: int,
        seed: Optional[int] = None
    ) -> tuple[DataFrame, dict[str, Any]]:
    """
    Generates a random instance following the Taillard method [1]. Processing times are randomly generated between 1 and 99 units,
    and the operations are randomly assigned to the machines, with each job having exactly one operation per machine.

    Parameters
    ----------
    n_jobs : int
        Number of jobs in the instance.

    n_machines : int
        Number of machines in the instance.

    seed : int, optional
        Seed for the random number generator.


    Returns
    -------
    instance : pandas.DataFrame
        DataFrame with the instance data in the following columns:
        - job: Job ID.
        - operation: Operation ID.
        - machine: Machine ID.
        - processing_time: Processing time for the operation.

    metadata : dict
        Dictionary with metadata about the instance.

    References
    ----------
    [1] Taillard, É. D. (1993). Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2), 278-285.
    """

    job_ids       = np.repeat(np.arange(n_jobs), n_machines)
    operation_ids = np.tile(np.arange(n_machines), n_jobs)

    rng = np.random.default_rng(seed)

    machine_ids      = rng.permuted(operation_ids.reshape(n_jobs, n_machines), axis=1).flatten()
    processing_times = rng.integers(1, 99, n_jobs * n_machines)

    instance = DataFrame({
        'job': job_ids,
        'operation': operation_ids,
        'machine': machine_ids,
        'processing_time': processing_times
    })

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    return instance, metadata

# TODO: Reference for the Demirkol instance generation method, p ~ U(1, 200)
def generate_demirkol_instance(
        n_jobs: int,
        n_machines: int,
        seed: Optional[int] = None
    ) -> tuple[DataFrame, dict[str, Any]]:
    job_ids       = np.repeat(np.arange(n_jobs), n_machines)
    operation_ids = np.tile(np.arange(n_machines), n_jobs)

    rng = np.random.default_rng(seed)

    machine_ids      = rng.permuted(operation_ids.reshape(n_jobs, n_machines), axis=1).flatten()
    processing_times = rng.integers(1, 200, n_jobs * n_machines)

    instance = DataFrame({
        'job': job_ids,
        'operation': operation_ids,
        'machine': machine_ids,
        'processing_time': processing_times
    })

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    return instance, metadata


def dirichlet_multinomial(rng: np.random.Generator, alpha: NDArray[np.floating[Any]], n: int) -> NDArray[np.int64]:
    return rng.multinomial(n, rng.dirichlet(alpha))


def generate_known_optimal_instance(
        n_ops: int,
        n_machines: int,
        optimal_makespan: int,
        seed: Optional[int] = None
    ) -> tuple[DataFrame, dict[str, Any]]:
    """
    Generates a random instance with known optimal makespan, following the method proposed by Da Col and Teppan [2].

    Parameters
    ----------
    n_ops : int
        Total number of operations in the instance.

    n_machines : int
        Number of machines in the instance.

    optimal_makespan : int
        Desired optimal makespan for the instance.

    seed : int, optional
        Seed for the random number generator.


    Returns
    -------
    instance : pandas.DataFrame
        DataFrame with the instance data in the following columns:
        - job: Job ID.
        - operation: Operation ID.
        - machine: Machine ID.
        - processing_time: Processing time for the operation.

    metadata : dict
        Dictionary with metadata about the instance.

    References
    ----------
    [2] Da Col, G., & Teppan, E. C. (2022). Industrial-size job shop scheduling with constraint programming. Operations Research Perspectives, 9, 100249.
    """

    instance = DataFrame(columns=['job', 'operation', 'machine', 'processing_time'], index=range(n_ops))

    rng = np.random.default_rng(seed)

    avg_op_per_machine  = n_ops // n_machines

    n_ops_per_machine = rng.normal(avg_op_per_machine, 0.05 * avg_op_per_machine, size=n_machines).astype(int)
    n_ops_per_machine[-1] = n_ops - n_ops_per_machine[:-1].sum()

    instance['machine']         = np.concatenate([np.full(n_ops_per_machine[machine], machine) for machine in range(n_machines)])
    instance['processing_time'] = np.concatenate([dirichlet_multinomial(rng, np.ones(n_ops_per_machine[machine]), optimal_makespan) for machine in range(n_machines)]).astype(int)

    start_times = (instance.groupby('machine')['processing_time'].cumsum() - instance['processing_time']).to_numpy()
    order       = np.argsort(start_times, stable=True)
    instance    = instance.loc[order]

    start_times = start_times[order]
    end_times   = start_times + instance['processing_time'].to_numpy()
    machines    = instance['machine'].to_numpy()
    chosen      = np.zeros(n_ops, dtype=bool)

    job_ids      = np.zeros(n_ops, dtype=int)
    operation_id = np.zeros(n_ops, dtype=int)

    job_machines = np.zeros((0, n_machines), dtype=bool)

    new_job_id = 0
    for i in range(n_ops):
        if not chosen[i]:
            job_ids[i]      = new_job_id
            operation_id[i] = 0
            chosen[i]       = True

            job_machines = np.vstack([job_machines, np.zeros(n_machines, dtype=bool)])
            job_machines[new_job_id, machines[i]] = True

            new_job_id += 1

        job_id = job_ids[i]
    
        possible_sucessors = (end_times[i] <= start_times[i+1:]) & ~job_machines[job_id][machines[i+1:]] & ~chosen[i+1:]

        if not possible_sucessors.any():
            continue

        possible_sucessors = np.where(possible_sucessors)[0]

        num_possible_sucessors = len(possible_sucessors)

        idx = np.clip(rng.poisson(1), 0, num_possible_sucessors-1)

        suc = int(possible_sucessors[idx] + i+1)

        job_ids[suc]      = job_id
        operation_id[suc] = operation_id[i] + 1
        chosen[suc]       = True

        job_machines[job_id, machines[suc]] = True


    instance['job']       = job_ids
    instance['operation'] = operation_id

    instance = instance.sort_values(['job', 'operation']).reset_index(drop=True)

    metadata = {
        'n_jobs'          : new_job_id,
        'n_machines'      : n_machines,
        'optimal_makespan': optimal_makespan
    }

    return instance, metadata
    


def generate_vepsalainen_instance(
        n_jobs: int,
        n_machines: int,
        mean_processing_time: int = 15,
        processing_time_range: int = 10,
        shop_type: Literal['uniform', 'proportionate', 'bottleneck'] = 'uniform',
        due_date_type: Literal['loose', 'tight'] = 'loose',
        utilization_level: float = 1.,
        seed: Optional[int] = None
    ) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)

    average_total_job_time = mean_processing_time * (n_machines + 1)/2

    arrival_rate = utilization_level * n_machines / average_total_job_time

    inter_arrival_times = rng.exponential(1/arrival_rate, size=n_jobs)
    arrival_times = np.cumsum(inter_arrival_times, dtype=int)

    num_operations = rng.integers(1, n_machines, size=n_jobs)
    total_operations = num_operations.sum()

    operation_id = np.concatenate([
        np.arange(n_ops) for n_ops in num_operations
    ])

    job_id = np.concatenate([
        np.full(n_ops, job) for job, n_ops in enumerate(num_operations)
    ])

    machines = np.concatenate([
        rng.permutation(n_machines)[:n_ops] for n_ops in num_operations
    ])

    min_processing_time = mean_processing_time - processing_time_range
    max_processing_time = mean_processing_time + processing_time_range

    if shop_type == 'proportionate':
        sizes = rng.integers(min_processing_time, max_processing_time, size=n_jobs)
        processing_times = np.concatenate([
            rng.integers(0.33 * size, 1.67 * size, size=n_ops) for size, n_ops in zip(sizes, num_operations)
        ])

    else:
        sizes = np.full(n_jobs, 15)
        processing_times = rng.integers(min_processing_time, max_processing_time, total_operations)

    # The bottleneck is not quite the same as the original paper. The original paper had the machines affected
    # by different rates: (0.7, O.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2), however due to the variable
    # number of machines in this implementation, we decided to make the bottleneck effect more pronounced, affecting
    # 66% of the machines, with 33% being faster and 33% being slower.
    if shop_type == 'bottleneck':
        bottleneck_machines = rng.choice(n_machines, int(0.66 * n_machines), replace=False)

        faster_machines = bottleneck_machines[:int(0.33 * n_machines)]
        slower_machines = bottleneck_machines[int(0.33 * n_machines):]

        processing_times[machines == faster_machines] *= 0.7 # 30% faster
        processing_times[machines == slower_machines] *= 1.2 # 20% slower

        processing_times = processing_times.astype(int)

    job_weights = rng.integers(1, 2*sizes)

    if due_date_type == 'tight':
        flow_allowance = rng.integers(0, 6*int(average_total_job_time), n_jobs)

    else:
        flow_allowance = rng.integers(0, 12*int(average_total_job_time), n_jobs)

    due_dates = arrival_times + flow_allowance

    instance = DataFrame({
        'job': job_id,
        'operation': operation_id,
        'machine': machines,
        'processing_time': processing_times,
    })

    job_information = DataFrame({
        'arrival_time': arrival_times,
        'due_date': due_dates,
        'weight': job_weights,
    })

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines,
    }

    return instance, job_information, metadata
