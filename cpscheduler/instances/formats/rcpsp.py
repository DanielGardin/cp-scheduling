r"""RCPSP instance format reader and writer.

The standard format has the following structure:
```
#n #r
(resource_capacity ){r}\n
(processing_time (resource_usage ){r} (successors )+\n){n}
```

where n is the number of tasks and r is the number of resources.
"""

from pathlib import Path
from typing import Any

InstanceReturnType = tuple[dict[str, list[Any]], dict[str, Any], dict[str, Any]]


def read_rcpsp_instance(
    path: Path | str,
) -> InstanceReturnType:
    r"""Read an RCPSP instance from a file.

    This file must be in the Patterson format, with the following structure:
    ```
    #n #r
    (resource_capacity ){r}\n
    (processing_time (resource_usage ){r} (successors )+\n){n}
    ```

    Parameters
    ----------
    path : str or Path
        Path to the file containing the instance data.

    Returns
    -------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - processing_time: Processing time for each task.
        - resource_{id}: Usage of each resource for each task.

    global_instance : dict[str, Any]
        Dictionary with the following keys:
        - capacity_{id}: Capacity of each resource.
        - precedence: Adjacency list of the precedence graph.

    metadata : dict[str, Any]
        Dictionary with metadata about the instance.
        - n_tasks: Number of tasks in the instance.
        - n_resources: Number of resources in the instance.

    """
    with open(path) as f:
        n_tasks, n_resources = map(int, f.readline().split())

        n_tasks -= 2  # Remove dummy tasks

        instance: dict[str, list[Any]] = {
            "processing_time": [],
            **{
                f"resource_{resource_id}": []
                for resource_id in range(n_resources)
            },
        }

        global_instance: dict[str, Any] = {
            f"capacity_{resource_id}": capacity
            for resource_id, capacity in enumerate(
                map(int, f.readline().split())
            )
        }

        # Dummy line
        f.readline()

        precedence: dict[int, list[int]] = {}

        for task_id in range(n_tasks):
            task_data = list(map(int, f.readline().split()))

            instance["processing_time"].append(task_data[0])

            for resource_id in range(n_resources):
                instance[f"resource_{resource_id}"].append(
                    task_data[resource_id + 1]
                )

            for task in task_data[n_resources + 2 :]:
                if task - 2 != n_tasks:
                    precedence.setdefault(task - 2, []).append(task_id)

        f.readline()  # Dummy line (Sink task)

        global_instance["precedence"] = precedence

        metadata = {
            "n_tasks": n_tasks,
            "n_resources": n_resources,
        }

        return instance, global_instance, metadata


def write_rcpsp_instance(
    instance: dict[str, list[Any]],
    global_instance: dict[str, Any],
    path: Path | str,
) -> None:
    r"""Write an RCPSP instance to a file.

    This file will be in the Patterson format, with the following structure:
    ```
    #n #r
    (resource_capacity ){r}\n
    (processing_time (resource_usage ){r} (successors )+\n){n}
    ```

    Parameters
    ----------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - processing_time: Processing time for each task.
        - resource_{id}: Usage of each resource for each task.

    global_instance : dict[str, Any]
        Dictionary with the following keys:
        - capacity_{id}: Capacity of each resource.
        - precedence: Adjacency list of the precedence graph.

    path : str or Path
        Path to the file where the instance data will be written.
    """
    n_tasks = len(instance["processing_time"])
    n_resources = len(global_instance) - 1  # Exclude precedence

    with open(path, "w") as f:
        f.write(f"{n_tasks + 2} {n_resources}\n")
        f.write(
            " ".join(
                str(global_instance[f"capacity_{resource_id}"])
                for resource_id in range(n_resources)
            )
            + "\n"
        )
        f.write("\n")  # Dummy line

        for task_id in range(n_tasks):
            processing_time = instance["processing_time"][task_id]
            resource_usages = [
                instance[f"resource_{resource_id}"][task_id]
                for resource_id in range(n_resources)
            ]
            successors = [
                str(successor + 2)
                for successor in global_instance["precedence"].get(task_id, [])
            ]

            f.write(
                f"{processing_time} {' '.join(map(str, resource_usages))} {' '.join(successors)}\n"
            )

        f.write("\n")  # Dummy line (Sink task)
