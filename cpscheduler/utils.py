from typing import Optional, Any
from numpy.typing import NDArray

from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd

def structured_to_dataframe(structured_array: NDArray[np.void], index_col: Optional[str] = None) -> pd.DataFrame:
    if index_col is not None:
        return pd.DataFrame(structured_array).set_index(index_col)

    return pd.DataFrame(structured_array)


def dataframe_to_structured(df: pd.DataFrame) -> NDArray[np.void]:
    structured_array = df.to_records(index=True)

    return np.asarray(structured_array)