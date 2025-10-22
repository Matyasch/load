from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ldp_github.ldp import LDP

from algorithms.load import is_explicit_ancestor, is_possible_ancestor
from algorithms.mb_by_mb import mb_by_mb_alg


def ldp_plus(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    **kwargs,
) -> dict:
    """
    LDP algorithm to identify a valid adjustment set for exposure-outcome pairs.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        **kwargs: Additional arguments.

    Returns:
        dict: Learned causal partitions.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    t1, t2 = targets
    G[t1] = mb_by_mb_alg(data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set)
    G[t2] = mb_by_mb_alg(data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set)

    data = pd.DataFrame(data)
    results = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        ldp_alg = LDP(data, ci_test)
        results[(t1, t2)] = ldp_alg.partition_z(t1, t2, alpha=alpha)
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        ldp_alg = LDP(data, ci_test)
        results[(t2, t1)] = ldp_alg.partition_z(t2, t1, alpha=alpha)
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            ldp_alg = LDP(data, ci_test)
            results[(t1, t2)] = ldp_alg.partition_z(t1, t2, alpha=alpha)
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            ldp_alg = LDP(data, ci_test)
            results[(t2, t1)] = ldp_alg.partition_z(t2, t1, alpha=alpha)
            ancestry[(t2, t1)] = "possible"

    return {"results": str(results), "ancestry": str(ancestry)}
