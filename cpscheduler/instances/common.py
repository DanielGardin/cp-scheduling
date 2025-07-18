from typing import Callable, Literal, Any
import random as rng

ScheduleSetup = Literal["parallel", "jobshop"]


def generate_instance(
    n_jobs: int,
    n_machines: int,
    processing_time_dist: Callable[[], int],
    release_time_dist: Callable[[], int] | None = None,
    due_date_dist: Callable[[], int] | None = None,
    slack_dist: Callable[[], int] | None = None,
    weight_dist: Callable[[], int | float] | None = None,
    setup: ScheduleSetup = "parallel",
    seed: int | None = None,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """
    Generates a random scheduling instance
    """
    rng.seed(seed)

    instance: dict[str, list[Any]] = {}
    job_instance: dict[str, list[Any]] = {}

    if setup == "parallel":
        instance["processing_time"] = [processing_time_dist() for _ in range(n_jobs)]

        if release_time_dist is not None:
            instance["release_time"] = [release_time_dist() for _ in range(n_jobs)]

        if due_date_dist is not None:
            instance["due_date"] = [due_date_dist() for _ in range(n_jobs)]

        elif slack_dist is not None:
            assert (
                release_time_dist is not None
            ), "Release time distribution must be provided if slack distribution is used"

            instance["due_date"] = [
                release_time + proc_time + slack_dist()
                for proc_time, release_time in zip(
                    instance["processing_time"], instance["release_time"]
                )
            ]

        if weight_dist is not None:
            instance["weight"] = [weight_dist() for _ in range(n_jobs)]

    elif setup == "jobshop":
        n_tasks = n_jobs * n_machines

        instance["job"] = [task_id // n_machines for task_id in range(n_tasks)]
        instance["operation"] = [task_id % n_machines for task_id in range(n_tasks)]
        instance["processing_time"] = [processing_time_dist() for _ in range(n_tasks)]

        machines: list[int] = []
        for job_id in range(n_jobs):
            machines.extend(rng.sample(range(n_machines), n_machines))

        instance["machine"] = machines

        if release_time_dist is not None:
            job_instance["release_time"] = [release_time_dist() for _ in range(n_jobs)]

        if due_date_dist is not None:
            job_instance["due_date"] = [due_date_dist() for _ in range(n_jobs)]

        elif slack_dist is not None:
            assert (
                release_time_dist is not None
            ), "Release time distribution must be provided if slack distribution is used"
            job: int
            processing_time: int

            due_dates: list[int] = [
                release_date + slack_dist()
                for release_date in job_instance["release_time"]
            ]
            for job, processing_time in zip(
                instance["job"], instance["processing_time"]
            ):
                due_dates[job] += processing_time

            job_instance["due_date"] = due_dates

        if weight_dist is not None:
            job_instance["weight"] = [weight_dist() for _ in range(n_jobs)]

    return instance, job_instance
