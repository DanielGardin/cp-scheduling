from pathlib import Path

import numpy as np
import pandas as pd

def read_instance(path: Path | str) -> pd.DataFrame:
    with open(path, 'r') as f:
        n_jobs, n_machines = map(int, f.readline().split())

        instance_data = np.array([
            list(map(int, line.split())) for line in f.readlines()
        ])

        instance = pd.DataFrame(columns=['job', 'operation', 'machine', 'processing_time'])

        instance['job']             = np.repeat(np.arange(n_jobs), n_machines)
        instance['operation']       = np.tile(np.arange(n_machines), n_jobs)
        instance['machine']         = instance_data[:, ::2].flatten()
        instance['processing_time'] = instance_data[:, 1::2].flatten()

    return instance