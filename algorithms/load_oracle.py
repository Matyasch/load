from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from algorithms.load import (
    get_neighbors,
    is_amenable,
    is_possible_ancestor,
)
from algorithms.mb_by_mb import mb_by_mb_alg, get_locally_valid_parent_sets, grow_shrink
from algorithms.cmb import cmb_alg


def load_oracle(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    treatment: int,
    outcome: int,
    lcd_algorithm: str = "mb_by_mb",
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
        lcd_algorithm (str): The local causal discovery algorithm to use.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        **kwargs: Additional keyword arguments are ignored.
    Returns:
        dict: Adjustment sets and a boolean indicating if the causal effect is identifiable.
    """
    if lcd_algorithm == "mb_by_mb":
        lcd_alg = mb_by_mb_alg
    elif lcd_algorithm == "cmb":
        lcd_alg = cmb_alg
    else:
        raise ValueError(f"Unknown local causal discovery algorithm: {lcd_algorithm}")

    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)
    adj_sets = dict()

    G[treatment] = lcd_alg(
        data, ci_test, alpha, treatment, mb_algorithm, MB, L, sep_set
    )

    # Check if amenable
    unoriented = get_neighbors(G[treatment], treatment).unoriented
    for v in unoriented:
        G[v] = lcd_alg(data, ci_test, alpha, v, mb_algorithm, MB, L, sep_set)
        if not is_amenable(treatment, outcome, v, G[v], ci_test, alpha):
            local_sets = get_locally_valid_parent_sets(G[treatment], treatment, outcome)
            adj_sets[(treatment, outcome)] = local_sets
            return {
                "adj_sets": str(adj_sets),
                "identifiable": False,
            }

    # Identify possible descendants of treatment
    pdesc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_possible_ancestor(treatment, v, G[treatment], ci_test, alpha):
            pdesc.add(v)

    # Obtain optimal adjustment set via forbidden projection
    forb = mb_by_mb_alg(data, ci_test, alpha, outcome, mb_algorithm, ignore=pdesc)
    oset = get_neighbors(forb, outcome).parents - {treatment}

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True}
