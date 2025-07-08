"""
    data.py

This module contains functions to handle data management for the constraints, setups and objectives
within the CPScheduler environment.
"""

from typing import Any
from typing_extensions import Self

from ._common import (
    ObsType,
    MACHINE_ID,
    TASK_ID,
    TIME,
)

from .utils import convert_to_list

JOB_ID_ALIASES = ["job", "job_id"]


class SchedulingData:
    "A class to hold static scheduling data for the CPScheduler environment."

    n_tasks: TASK_ID
    n_jobs: TASK_ID
    n_machines: MACHINE_ID

    task_data: dict[str, list[Any]]
    jobs_data: dict[str, list[Any]]

    job_ids: list[TASK_ID]

    def __init__(
        self,
        task_data: dict[str, list[Any]],
        processing_times: list[dict[MACHINE_ID, TIME]],
        jobs_data: dict[str, list[Any]],
        job_feature: str = "",
    ) -> None:
        self.task_data = task_data.copy()
        self.jobs_data = jobs_data.copy()
        self.processing_times = processing_times.copy()

        self.n_tasks = TASK_ID(len(processing_times))
        for feature, values in self.task_data.items():
            length = TASK_ID(len(values))

            if length != self.n_tasks:
                raise ValueError(
                    f"Feature '{feature}' must have length equal to the number of tasks,"
                    f"expected {self.n_tasks}, got {len(values)}."
                )

        machines: set[MACHINE_ID] = set()
        for processing_time in processing_times:
            machines.update(processing_time.keys())

        self.n_machines = MACHINE_ID(len(machines))

        if not job_feature:
            for alias in JOB_ID_ALIASES:
                if alias in self.task_data:
                    job_feature = alias

        if job_feature in self.task_data:
            self.job_ids = convert_to_list(self.task_data.pop(job_feature), TASK_ID)
            self.n_jobs = TASK_ID(len(set(self.job_ids)))

        else:  # If no job feature is provided, we assume each task is its own job
            self.job_ids = [task_id for task_id in range(self.n_tasks)]
            self.n_jobs = self.n_tasks

    @classmethod
    def empty(cls) -> Self:
        "Create an empty SchedulingData instance."
        return cls(
            task_data={"job_id": []},
            jobs_data={"job_id": []},
            processing_times=[],
        )

    def __getitem__(self, key: str) -> list[Any]:
        "Get a specific data feature for all tasks or jobs."
        if key in self.task_data:
            return self.task_data[key]

        if key in self.jobs_data:
            return self.jobs_data[key]

        raise KeyError(f"Feature '{key}' not found in tasks or jobs data.")

    def get_task_level_data(self, feature: str) -> list[Any]:
        "Get a specific, task or job, data feature for all tasks."
        if feature in self.task_data:
            return self.task_data[feature]

        if feature in self.jobs_data:
            job_data = self.jobs_data[feature]

            return [job_data[job_id] for job_id in self.task_data["job_id"]]

        raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

    def get_job_level_data(self, feature: str) -> list[Any]:
        "Get a specific job data feature for all jobs."
        if feature in self.jobs_data:
            return self.jobs_data[feature]

        if feature in self.task_data:
            if self.n_tasks == self.n_jobs:
                return self.task_data[feature]

            job_level_data: list[Any] = [None for _ in range(self.n_jobs)]
            for task_id, job_id in enumerate(self.task_data["job_id"]):
                job_level_data[job_id] = self.task_data[feature][task_id]

            return job_level_data

        raise KeyError(f"Feature '{feature}' not found in jobs data.")

    def add_data(self, feature: str, values: list[Any]) -> None:
        length = TASK_ID(len(values))

        if length == self.n_tasks:
            if feature in self.task_data:
                return

            self.task_data[feature] = values

        elif length == self.n_jobs:
            if feature in self.jobs_data:
                return

            self.jobs_data[feature] = values

        else:
            raise ValueError(
                f"Feature '{feature}' must have length equal to the number of tasks or jobs,"
                f"expected {self.n_tasks} or {self.n_jobs}, got {length}."
            )

    def export_state(self) -> ObsType:
        "Export the status of the tasks in a dictionary format."
        task_state = {
            "task_id": [task_id for task_id in range(self.n_tasks)],
            "job_id": self.job_ids.copy(),
            **{feature: self.task_data[feature] for feature in self.task_data},
        }

        job_state = {
            "job_id": [job_id for job_id in range(self.n_jobs)],
            **{feature: self.jobs_data[feature] for feature in self.jobs_data},
        }

        return task_state, job_state
