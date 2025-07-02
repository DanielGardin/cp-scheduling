from pathlib import Path

from typing import Any

def read_rcpsp_instance(path: Path | str) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """
    Reads an instance from a file. The file must be in the Patterson format, with the following structure:
    - The first line contains the number of jobs and the number of resources in the instance.
    - The second line contains the resource capacities.
    - The following lines contain the processing times, the amount of
    resource required and successors for each task.

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
        - processing_time: Processing time for the operation.
        - resource: Resource ID.

    metadata : dict
        Dictionary with metadata about the instance.
        - n_tasks: Number of tasks in the instance.
        - n_resources: Number of resources in the instance.
        - resource_capacities: List with the capacities of each resource.
        - successors: List with the successors of each task.
    """
    with open(path, "r") as f:
        n_tasks, n_resources = map(int, f.readline().split())

        n_tasks -= 2  # Remove dummy tasks

        resource_capacities = list(map(int, f.readline().split()))

        # Dummy line
        f.readline()

        durations = [0 for _ in range(n_tasks)]
        resource_demands = [[0 for _ in range(n_tasks)] for _ in range(n_resources)]

        precedence_tasks: dict[int, list[int]] = {}

        for task_id in range(n_tasks):
            task_data = list(map(int, f.readline().split()))

            durations[task_id] = task_data[0]

            for resource_id in range(n_resources):
                resource_demands[resource_id][task_id] = task_data[resource_id + 1]

            precedence_tasks[task_id] = [
                task - 2 for task in task_data[n_resources + 2 :] if task - 2 != n_tasks
            ]

        f.readline()  # Dummy line (Sink task)

        instance = {
            "processing_time": durations,
            **{
                f"resource_{resource_id}": resource_demands[resource_id]
                for resource_id in range(n_resources)
            },
        }

        metadata = {
            "n_tasks": n_tasks,
            "n_resources": n_resources,
            "resource_capacities": resource_capacities,
            "precedence_tasks": precedence_tasks,
        }

        metadata_line = (
            f.readline()
        )  # Skip the first line that separates the data from the metadata

        print(metadata_line)

        while metadata_line:
            metadata_line = f.readline()

            key, value = metadata_line.split(":")

            metadata[key.strip()] = value.strip()

    return instance, metadata