# Purpose: Provide lightweight parquet loading and stored transition-matrix decoding.
# Public API: ParquetReader.read and unpack_matrix.
# Consumers: Available to representation consumers; no direct in-repository callers yet.

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# Purpose: Bind a parquet path without loading its contents during construction.
# Consumers: Callers opting into the shared representation reader utility.
class ParquetReader:
    # Purpose: Retain the caller's path unchanged until an explicit read.
    # Callers: Representation consumers constructing a ParquetReader.
    def __init__(self, path: Path) -> None:
        self.path = path

    # Purpose: Read the parquet frame and let Polars report file or schema failures.
    # Callers: Representation consumers using this reader; no caching or mutation occurs.
    def read(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)


# Purpose: Decode a stored flat transition vector into the caller-supplied matrix shape.
# Callers: Representation consumers; NumPy raises for incompatible shapes or values.
def unpack_matrix(row: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    flat = row["transition_probabilities_flat"]
    return np.array(flat, dtype=np.float32).reshape(shape)
