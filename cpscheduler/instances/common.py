from typing import Any, Optional, Callable, Literal
from pandas import DataFrame

import random as rng

ScheduleSetup = Literal["parallel", "jobshop"]

def generate_instance(
    n_jobs: int,
    n_machines: int,
    processing_time_dist: Callable[[], int],
    release_time_dist: Optional[Callable[[], int]] = None,
    due_date_dist: Optional[Callable[[], int]] = None,
    slack_dist: Optional[Callable[[], int]] = None,
    weight_dist: Optional[Callable[[], int | float]] = None,
    setup: ScheduleSetup = "parallel",
    seed: Optional[int] = None,
) -> tuple[DataFrame, DataFrame]:
    """
    Generates a random scheduling instance
    """
    rng.seed(seed)

    instance = DataFrame()
    job_instance = DataFrame()
    
    if setup == "parallel":
        instance["processing_time"] = [processing_time_dist() for _ in range(n_jobs)]
    
        if release_time_dist is not None:
            instance["release_time"] = [release_time_dist() for _ in range(n_jobs)]

        if due_date_dist is not None:
            instance["due_date"] = [due_date_dist() for _ in range(n_jobs)]
        
        elif slack_dist is not None:
            assert release_time_dist is not None, "Release time distribution must be provided if slack distribution is used"

            instance["due_date"] = [
                release_time + proc_time + slack_dist() for proc_time, release_time in zip(instance["processing_time"], instance["release_time"])
            ]
        
        if weight_dist is not None:
            instance["weight"] = [weight_dist() for _ in range(n_jobs)]
        
    elif setup == "jobshop":
        n_tasks = n_jobs * n_machines

        instance["job"]             = [task_id // n_machines for task_id in range(n_tasks)]
        instance["operation"]       = [task_id % n_machines for task_id in range(n_tasks)]
        instance["machine"]         = sum([rng.sample(range(n_machines), n_machines) for _ in range(n_jobs)], [])
        instance["processing_time"] = [processing_time_dist() for _ in range(n_tasks)]

        if release_time_dist is not None:
            job_instance["release_time"] = [release_time_dist() for _ in range(n_jobs)]

        if due_date_dist is not None:
            job_instance["due_date"] = [due_date_dist() for _ in range(n_jobs)]
        
        elif slack_dist is not None:
            assert release_time_dist is not None, "Release time distribution must be provided if slack distribution is used"

            job_instance["due_date"] = [
                release_time + sum(instance.loc[instance["job"] == job_id, "processing_time"]) + slack_dist() for job_id, release_time in enumerate(job_instance["release_time"])
            ]
        
        if weight_dist is not None:
            job_instance["weight"] = [weight_dist() for _ in range(n_jobs)]


    return instance, job_instance