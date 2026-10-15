# Purpose: Provide lightweight assertions for transition tensor shape and row sums.
# Public API: assert_stochastic and assert_shape.
# Consumers: Available to representation callers; no direct in-repository callers yet.

import numpy as np


# Purpose: Require positive-sum rows to be near one under NumPy allclose tolerances.
# Callers: Representation consumers opting into row-sum checks.
# Nonpositive rows are skipped; this does not validate individual entries or finiteness.
# Python assertions can be disabled with optimized execution.
def assert_stochastic(matrix: np.ndarray, atol: float = 1e-6) -> None:
    row_sums = matrix.sum(axis=-1)
    observed = row_sums > 0
    assert np.allclose(row_sums[observed], 1.0, atol=atol), "non-stochastic rows"


# Purpose: Assert exact tensor dimensions without reshaping or mutating the input.
# Callers: Representation consumers opting into shape checks.
# Python assertions can be disabled with optimized execution.
def assert_shape(matrix: np.ndarray, expected: tuple[int, ...]) -> None:
    assert matrix.shape == expected, f"shape mismatch: {matrix.shape} != {expected}"
