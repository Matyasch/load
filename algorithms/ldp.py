from typing import Callable

import numpy as np
import pandas as pd

from ldp_github.ldp import LDP


def ldp(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    treatment: int,
    outcome: int,
    **kwargs,
) -> dict:
    """
    LDP algorithm to identify a valid adjustment set for exposure-outcome pairs.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        **kwargs: Additional arguments.

    Returns:
        dict: Learned causal partitions.
    """
    data = pd.DataFrame(data)
    result = LDP(data, ci_test).partition_z(treatment, outcome, alpha=alpha)
    return {"results": str({(treatment, outcome): result})}
