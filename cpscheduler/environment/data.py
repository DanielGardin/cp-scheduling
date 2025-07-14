"""
    data.py

This module contains functions to handle data management for the constraints, setups and objectives
within the CPScheduler environment.
"""

from typing import Any
from typing_extensions import Self

from ._common import ObsType, MACHINE_ID, TASK_ID, TIME, InstanceConfig

from .utils import convert_to_list

JOB_ID_ALIASES = ["job", "job_id", "jobs", "jobs_ids"]


class SchedulingData:
    "A class to hold static scheduling data for the CPScheduler environment."

    n_tasks: TASK_ID
    n_jobs: TASK_ID
    n_machines: MACHINE_ID

    safe_converse: bool
    "Whether task ID and job ID can be safely converted to each other."

    task_data: dict[str, list[Any]]
    jobs_data: dict[str, list[Any]]

    job_ids: list[TASK_ID]

    alias: dict[str, str]

    def __init__(
        self,
        task_data: dict[str, list[Any]],
        jobs_data: dict[str, list[Any]],
        job_feature: str = "",
    ) -> None:
        self.alias = {}

        self.task_data = task_data.copy()
        self.jobs_data = jobs_data.copy()

        self.n_tasks = TASK_ID(len(next(iter(self.task_data.values()))))
        for feature, values in self.task_data.items():
            length = TASK_ID(len(values))

            if length != self.n_tasks:
                raise ValueError(
                    f"Feature '{feature}' must have length equal to the number of tasks,"
                    f"expected {self.n_tasks}, got {len(values)}."
                )

        if not job_feature:
            for alias in JOB_ID_ALIASES:
                if alias in self.task_data:
                    job_feature = alias

        if job_feature in self.task_data:
            self.job_ids = convert_to_list(self.task_data.pop(job_feature), TASK_ID)
            self.n_jobs = TASK_ID(len(set(self.job_ids)))

            self.safe_converse = self.n_tasks == self.n_jobs and all(
                job_id == task_id for job_id, task_id in enumerate(self.job_ids)
            )

        else:  # If no job feature is provided, we assume each task is its own job
            self.job_ids = [task_id for task_id in range(self.n_tasks)]
            self.n_jobs = self.n_tasks

            self.safe_converse = True

    def import_machine_data(
        self,
        processing_times: list[dict[MACHINE_ID, TIME]],
        machine_data: dict[str, list[Any]],
    ) -> None:
        """
        Import machine-specific data into the scheduling data.

        Parameters
        ----------
        processing_times: list[dict[MACHINE_ID, TIME]]
            The processing times for each task on each machine.
        machine_data: dict[str, list[Any]]
            Additional machine-specific data to be added.
        """
        self.processing_times = processing_times

        machines: set[MACHINE_ID] = set()
        for processing_time in processing_times:
            machines.update(processing_time.keys())

        self.n_machines = MACHINE_ID(len(machines))

    @classmethod
    def empty(cls) -> Self:
        "Create an empty SchedulingData instance."
        return cls(
            task_data={},
            jobs_data={},
        )

    def add_alias(self, alias: str, feature: str) -> None:
        if feature not in self.task_data and feature not in self.jobs_data:
            raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

        if alias != feature:
            self.alias[alias] = feature

    def __getitem__(self, feature: str) -> list[Any]:
        "Get a specific data feature for all tasks or jobs."
        if feature in self.alias:
            feature = self.alias[feature]

        if feature in self.task_data:
            return self.task_data[feature]

        if feature in self.jobs_data:
            return self.jobs_data[feature]

        raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

    def get_task_data(self, feature: str, task_id: TASK_ID, default: Any = None) -> Any:
        "Get a specific task data feature for a given task."
        if feature in self.alias:
            feature = self.alias[feature]

        if feature in self.task_data:
            return self.task_data[feature][task_id]

        if feature in self.jobs_data:
            job_data = self.jobs_data[feature]
            job_id = self.job_ids[task_id]

            return job_data[job_id]

        if default is not None:
            return default

        raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

    def get_job_data(self, feature: str, job_id: TASK_ID, default: Any = None) -> Any:
        "Get a specific job data feature for a given job."
        if feature in self.alias:
            feature = self.alias[feature]

        if feature in self.jobs_data:
            return self.jobs_data[feature][job_id]

        if feature in self.task_data:
            if self.safe_converse:
                return self.task_data[feature][job_id]

            job_data_point: Any = None
            task_level_data = self.task_data[feature]
            for task_id, job in enumerate(self.job_ids):
                if job_id != job:
                    continue

                if job_data_point is None:
                    job_data_point = task_level_data[task_id]

                if job_data_point != task_level_data[task_id]:
                    raise ValueError(
                        f"Feature '{feature}' has inconsistent values for job {job_id}."
                    )

            return job_data_point

        if default is not None:
            return default

        raise KeyError(f"Feature '{feature}' not found in jobs data.")

    def get_task_level_data(self, feature: str) -> list[Any]:
        "Get a specific, task or job, data feature for all tasks."
        if feature in self.alias:
            feature = self.alias[feature]

        if feature in self.task_data:
            return self.task_data[feature]

        if feature in self.jobs_data:
            job_data = self.jobs_data[feature]

            return [job_data[job_id] for job_id in self.job_ids]

        raise KeyError(f"Feature '{feature}' not found in tasks or jobs data.")

    def get_job_level_data(self, feature: str) -> list[Any]:
        "Get a specific job data feature for all jobs."
        if feature in self.alias:
            feature = self.alias[feature]

        if feature in self.jobs_data:
            return self.jobs_data[feature]

        if feature in self.task_data:
            if self.safe_converse:
                return self.task_data[feature]

            task_level_data = self.task_data[feature]
            job_level_data: list[Any] = [None for _ in range(self.n_jobs)]
            for task_id, job_id in enumerate(self.job_ids):
                job_data_point = task_level_data[task_id]

                if job_level_data[job_id] is None:
                    job_level_data[job_id] = job_data_point

                if job_level_data[job_id] != job_data_point:
                    raise ValueError(
                        f"Feature '{feature}' has inconsistent values for job {job_id}."
                    )

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
            "job_id": self.job_ids,
            **self.task_data,
        }

        job_state = {
            "job_id": [job_id for job_id in range(self.n_jobs)],
            **self.jobs_data,
        }

        return task_state, job_state

    def to_dict(self) -> InstanceConfig:
        "Convert the scheduling data back to the instance format."
        instance_config: InstanceConfig = {}

        instance = {
            **self.task_data,
            "job_id": self.job_ids,
        }

        instance_config["instance"] = instance
        instance_config["processing_times"] = self.processing_times

        if len(self.jobs_data) > 1:
            instance_config["job_instance"] = {
                **self.jobs_data,
            }

        return instance_config
