r"""Da Col Teppan job shop instance reader and writer.

The Da Col Teppan format has the following structure:
```
#n #m
((machine_id processing_time)+ -1 -1\n){n}
```

where n is the number of jobs and m is the number of machines.
"""

from pathlib import Path
from typing import Any

InstanceReturnType = tuple[dict[str, list[Any]], dict[str, Any]]


def read_dacolteppan_jobshop_instance(path: str | Path) -> InstanceReturnType:
    r"""Read a job shop instance in Da Col Teppan format.

    This format is an extension of the standard job shop format, where each
    line corresponds to a job, but the number of operations per job can vary.
    The format has the following structure:
    ```
    #n #m
    ((machine_id processing_time)+ -1 -1\n){n}
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
            values = list(map(int, line.split()))

            n_operations = (len(values) - 2) // 2

            instance["job"].extend([job_id] * n_operations)
            instance["operation"].extend(list(range(n_operations)))
            instance["machine"].extend(values[: n_operations * 2 : 2])
            instance["processing_time"].extend(values[1 : n_operations * 2 : 2])

        metadata = {
            "n_jobs": n_jobs,
            "n_machines": n_machines,
        }

    return instance, metadata


def write_dacolteppan_jobshop_instance(
    instance: dict[str, list[Any]], path: str | Path
) -> None:
    r"""Write a job shop instance in Da Col Teppan format.

    The Da Col Teppan format has the following structure:
    ```
    #n #m
    ((machine_id processing_time)+ -1 -1\n){n}
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
        Path to the file where the instance will be written.

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

            for operation_id in range(len(operations)):
                machine_id, processing_time = operations[operation_id]
                f.write(f"{machine_id} {processing_time} ")

            f.write("-1 -1\n")
