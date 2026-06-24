r"""Taillard job shop instance reader and writer.

The Taillard format has the following structure:
```
#n #m
((machine ){m}\n){n}
(processing_time {m}\n){n}
```

where n is the number of jobs and m is the number of machines.
"""

from pathlib import Path
from typing import Any

InstanceReturnType = tuple[dict[str, list[Any]], dict[str, Any]]


def read_taillard_jobshop_instance(path: str | Path) -> InstanceReturnType:
    r"""Read a job shop instance in Taillard format.

    The Taillard format is a standard format for job shop instances, where each
    line corresponds to a job, and each job has the same number of operations.
    The format has the following structure:
    ```
    #n #m
    ((machine ){m}\n){n}
    (processing_time {m}\n){n}
    ```

    where n is the number of jobs and m is the number of machines.

    Parameters
    ----------
    path : str or Path
        Path to the file containing the instance data.

    Returns
    -------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - "job": List of job IDs for each task.
        - "operation": List of operation IDs for each task.
        - "machine": List of machine IDs for each task.
        - "processing_time": List of processing times for each task.

    metadata : dict[str, Any]
        Dictionary with metadata about the instance.
        Metadata keys can include:
        - "n_jobs": Number of jobs in the instance.
        - "n_machines": Number of machines in the instance.

    """
    with open(path) as f:
        n_jobs, n_machines = map(int, f.readline().strip().split())

        instance: dict[str, list[Any]] = {
            "job": [],
            "operation": [],
            "machine": [],
            "processing_time": [],
        }

        for job_id in range(n_jobs):
            line = f.readline().strip()
            machine_ids = list(map(int, line.split()))

            instance["job"].extend([job_id] * n_machines)
            instance["operation"].extend(range(n_machines))
            instance["machine"].extend(machine_ids)

        for _ in range(n_jobs):
            line = f.readline().strip()
            processing_times = list(map(int, line.split()))

            instance["processing_time"].extend(processing_times)

    metadata = {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
    }

    return instance, metadata


def write_taillard_jobshop_instance(
    instance: dict[str, list[Any]],
    path: str | Path,
) -> None:
    r"""Write a job shop instance in Taillard format.

    The Taillard format is a standard format for job shop instances, where each
    line corresponds to a job, and each job has the same number of operations.
    The format has the following structure:
    ```
    #n #m
    ((machine ){m}\n){n}
    (processing_time {m}\n){n}
    ```

    where n is the number of jobs and m is the number of machines.

    Parameters
    ----------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - "job": List of job IDs for each task.
        - "operation": List of operation IDs for each task.
        - "machine": List of machine IDs for each task.
        - "processing_time": List of processing times for each task.

    path : str or Path
        Path to the file where the instance data will be written.

    """
    n_jobs = max(instance["job"]) + 1
    n_machines = max(instance["machine"]) + 1

    with open(path, "w") as f:
        f.write(f"{n_jobs} {n_machines}\n")

        task_info: dict[int, dict[int, tuple[int, int]]] = {
            job_id: {} for job_id in range(n_jobs)
        }

        for i in range(len(instance["job"])):
            job_id = instance["job"][i]
            operation_id = instance["operation"][i]

            processing_time = instance["processing_time"][i]
            machine_id = instance["machine"][i]

            task_info[job_id][operation_id] = (machine_id, processing_time)

        for job_id in range(n_jobs):
            operations = task_info[job_id]

            for operation_id in range(n_machines):
                machine_id, _ = operations[operation_id]
                f.write(f"{machine_id} ")
            f.write("\n")

        for job_id in range(n_jobs):
            operations = task_info[job_id]

            for operation_id in range(n_machines):
                _, processing_time = operations[operation_id]
                f.write(f"{processing_time} ")
            f.write("\n")
