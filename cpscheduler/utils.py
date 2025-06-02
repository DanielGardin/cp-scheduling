from typing import Optional, Any
from numpy.typing import NDArray

from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd

def read_instance(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    with open(path, 'r') as f:
        n_jobs, n_machines = map(int, f.readline().split())

        # TODO: Allow metadata to be read from the bottom of the file, after the instance data

        instance_data = np.array([
            list(map(int, line.split())) for line in f.readlines()
        ])

        instance = pd.DataFrame(columns=['job', 'operation', 'machine', 'processing_time'])

        instance['job']             = np.repeat(np.arange(n_jobs), n_machines)
        instance['operation']       = np.tile(np.arange(n_machines), n_jobs)
        instance['machine']         = instance_data[:, ::2].flatten()
        instance['processing_time'] = instance_data[:, 1::2].flatten()

    metadata = {
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    return instance, metadata


def structured_to_dataframe(structured_array: NDArray[np.void], index_col: Optional[str] = None) -> pd.DataFrame:
    if index_col is not None:
        return pd.DataFrame(structured_array).set_index(index_col)

    return pd.DataFrame(structured_array)


def dataframe_to_structured(df: pd.DataFrame) -> NDArray[np.void]:
    structured_array = df.to_records(index=True)

    return np.asarray(structured_array)