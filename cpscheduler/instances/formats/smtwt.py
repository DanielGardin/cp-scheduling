r"""SMTWT instance format.

The standard format has the following structure:
```
#n
(processing_time weight due_date\n){n}
```

where n is the number of jobs.
"""

from pathlib import Path
from typing import Any

InstanceReturnType = tuple[dict[str, list[Any]], dict[str, Any]]


def read_smtwt_instance(
    path: Path | str,
) -> InstanceReturnType:
    r"""Read an SMTWT instance from a file.

    This file must be in the SMTWT format, with the following structure:
    ```
    #n
    (processing_time weight due_date\n){n}
    ```

    Parameters
    ----------
    path : str or Path
        Path to the file containing the instance data.

    Returns
    -------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - processing_time: Processing time of each task.
        - weight: Weight of each task.
        - due_date: Due date of each task.

    metadata : dict[str, Any]
        Dictionary with metadata about the instance.
        Metadata keys can include:
        - n_jobs: Number of jobs in the instance.

    """
    with open(path) as f:
        n_jobs = int(f.readline().strip())

        instance: dict[str, list[int]] = {
            "processing_time": [],
            "weight": [],
            "due_date": [],
        }

        for _ in range(n_jobs):
            line = f.readline().strip()

            processing_time, weight, due_date = map(int, line.split())
            instance["processing_time"].append(processing_time)
            instance["weight"].append(weight)
            instance["due_date"].append(due_date)

        metadata = {"n_jobs": n_jobs}

    return instance, metadata


def write_smtwt_instance(
    instance: dict[str, list[Any]],
    path: Path | str,
) -> None:
    r"""Write an SMTWT instance to a file.

    This file will be in the SMTWT format, with the following structure:
    ```
    #n
    (processing_time weight due_date\n){n}
    ```

    Parameters
    ----------
    instance : dict[str, list[Any]]
        Dictionary with the following keys:
        - processing_time: Processing time of each task.
        - weight: Weight of each task.
        - due_date: Due date of each task.

    path : str or Path
        Path to the file where the instance data will be written.

    """
    n_jobs = len(instance["processing_time"])

    with open(path, "w") as f:
        f.write(f"{n_jobs}\n")

        for i in range(n_jobs):
            processing_time = instance["processing_time"][i]
            weight = instance["weight"][i]
            due_date = instance["due_date"][i]

            f.write(f"{processing_time} {weight} {due_date}\n")
