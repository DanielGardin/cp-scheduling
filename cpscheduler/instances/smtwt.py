from pathlib import Path

from typing import Any

import random as rng

from .common import generate_instance


def read_smtwt_instance(
    path: Path | str,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """
    Reads an instance from a file. The file must be in the SMTWT format, with the following structure:
    - The first line contains the number of jobs.
    - The following lines are processed in the following order:
        - The first number is the processing time.
        - The next number is the weight.
        - The next number is the due date.

    Parameters
    ----------
    path : str or Path
        Path to the file containing the instance data.

    Returns
    -------
    instance : pandas.DataFrame
        DataFrame with the instance data in the following columns:
        - processing_time: Processing time of the task.
        - weight: Weight of the task.
        - due_date: Due date of the task.

    metadata : dict
        Dictionary with metadata about the instance.
    """
    with open(path, "r") as f:
        n_jobs = int(f.readline().strip())

        instance: dict[str, list[int]] = {
            "processing_time": [],
            "weight": [],
            "due_date": [],
        }

        for task_id in range(n_jobs):
            line = f.readline().strip()

            if line == "---":
                break
            processing_time, weight, due_date = map(int, line.split())
            instance["processing_time"].append(processing_time)
            instance["weight"].append(weight)
            instance["due_date"].append(due_date)

        metadata = {}

        f.readline()  # Skip the "---" line

        while True:
            metadata_line = f.readline()

            if not metadata_line:
                break

            key, value = metadata_line.split(": ")

            metadata[key.strip()] = value.strip()

    return instance, metadata


def generate_chu_instance(
    n_jobs: int,
    max_processing_time: int = 100,
    max_weight: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, list[Any]]:
    instance, _ = generate_instance(
        n_jobs,
        1,
        processing_time_dist=lambda: rng.randint(1, max_processing_time),
        weight_dist=lambda: rng.randint(1, max_weight),
    )

    total_processing_time = sum(instance["processing_time"])

    max_release_time = int(alpha * total_processing_time)
    max_slack_time = int(beta * total_processing_time)

    instance["release_time"] = [rng.randint(0, max_release_time) for _ in range(n_jobs)]
    instance["due_date"] = [
        release_date + processing_time + rng.randint(0, max_slack_time)
        for release_date, processing_time in zip(
            instance["release_time"], instance["processing_time"]
        )
    ]

    return instance
