from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from algorithms.load import is_explicit_ancestor, is_possible_ancestor
from algorithms.mb_by_mb import mb_by_mb_alg, get_locally_valid_parent_sets


def mb_by_mb_plus(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    **kwargs,
) -> dict:
    """
    MB-by-MB algorithm with extra steps to determine causal relationships.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Locally valid parent adjustment sets and learned causal relationships.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    t1, t2 = targets
    G[t1] = mb_by_mb_alg(data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set)
    G[t2] = mb_by_mb_alg(data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set)

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1, t2)
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2, t1)
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1, t2)
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2, t1)
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}
