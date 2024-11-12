from typing import Any
from pandas import DataFrame

from pathlib import Path


def read_instance(path: Path | str) -> tuple[DataFrame, dict[str, Any]]:
    with open(path, 'r') as f:
        n_jobs, n_machines = map(int, f.readline().split())

        # TODO: Allow metadata to be read from the file

        data: list[list[int]] = []

        for job_id in range(n_jobs):
            job_info = f.readline()
            
            job_data = list(map(int, job_info.split()))

            for operation_id, (machine_id, processing_time) in enumerate(zip(job_data[::2], job_data[1::2])):
                data.append([job_id, operation_id, machine_id, processing_time])


        instance = DataFrame(data, columns=['job', 'operation', 'machine', 'processing_time'])

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    return instance, metadata


# TODO: Implement a function for generating random taillard instances [1] and Known-optimal instances [2].
# [1] Taillard, É. D. (1993). Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2), 278-285.
# [2] Da Col