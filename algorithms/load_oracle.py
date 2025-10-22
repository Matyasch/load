from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from algorithms.load import (
    get_locally_valid_parent_sets,
    get_neighbors,
    is_amenable,
    is_explicit_ancestor,
)
from algorithms.mb_by_mb import mb_by_mb_alg


def load_oracle(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    treatment: int,
    outcome: int,
    mb_algorithm: str = "grow_shrink",
    **kwargs,
) -> dict:
    """
    Optimal adjustment set discovery using local causal discovery algorithms.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        treatment (int): The treatment node.
        outcome (int): The outcome node.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.

    Returns:
        dict: Adjustment sets and a boolean indicating if the causal effect is identifiable.
    """

    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    adj_sets = dict()

    G[treatment] = mb_by_mb_alg(
        data, ci_test, alpha, treatment, mb_algorithm, MB, L, sep_set
    )

    # Check if amenable
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            adj_sets[(treatment, outcome)] = get_locally_valid_parent_sets(
                G[treatment], treatment
            )
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
                "id_tests": ci_test.get_tests_per_order().tolist(),
            }
    id_tests = ci_test.get_tests_per_order().tolist()

    # Identify explicit descendants of treatment
    desc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_explicit_ancestor(treatment, v, G[treatment], ci_test, alpha):
            desc.add(v)
            G[v] = mb_by_mb_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)

    # Identify explicit mediators between treatment and outcome
    meds = set()
    for v in desc:
        if is_explicit_ancestor(v, outcome, G[v], ci_test, alpha):
            meds.add(v)

    # Identify optimal adjustment set
    G[outcome] = mb_by_mb_alg(
        data, ci_test, alpha, outcome, mb_algorithm, MB, L, sep_set
    )
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
    oset -= meds | {treatment}

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}
