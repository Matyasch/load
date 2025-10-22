from typing import Callable, Sequence

import numpy as np

from algorithms.ldecc import get_all_combinations, ldecc_alg


def is_explicit_ancestor(
    x: int,
    y: int,
    result: dict,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is an explicit ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        result (dict): Local information of x.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.

    Returns:
        bool: True if x is an explicit ancestor of y, False otherwise.
    """
    if y in result["children"]:
        return True
    elif y in result["parents"] | result["unoriented"]:
        return False
    return ci_test(x, y, result["parents"] | result["unoriented"]) < alpha


def is_possible_ancestor(
    x: int,
    y: int,
    result: dict,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is a possible ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        result (dict): Local information of x.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if x is a possible ancestor of y, False otherwise.
    """
    if y in result["children"] | result["unoriented"]:
        return True
    elif y in result["parents"]:
        return False
    return ci_test(x, y, result["parents"]) < alpha


def ldecc_plus(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: Sequence[int],
    **kwargs,
) -> dict:
    """
    LDECC algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned causal relationships.
    """
    t1, t2 = targets
    res_t1 = ldecc_alg(data, ci_test, alpha, t1)
    res_t2 = ldecc_alg(data, ci_test, alpha, t2)

    adj_sets = dict()
    ancestry = dict()
    # Determine causal relationships
    if is_explicit_ancestor(t1, t2, res_t1, ci_test, alpha):
        combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
        adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
        ancestry[(t1, t2)] = "explicit"
    elif is_explicit_ancestor(t2, t1, res_t2, ci_test, alpha):
        combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
        adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
        ancestry[(t2, t1)] = "explicit"
    else:
        if is_possible_ancestor(t1, t2, res_t1, ci_test, alpha):
            combs = get_all_combinations(res_t1["unoriented"], res_t1["non_colliders"])
            adj_sets[(t1, t2)] = [set(s) | res_t1["parents"] for s in combs]
            ancestry[(t1, t2)] = "possible"
        if is_possible_ancestor(t2, t1, res_t2, ci_test, alpha):
            combs = get_all_combinations(res_t2["unoriented"], res_t2["non_colliders"])
            adj_sets[(t2, t1)] = [set(s) | res_t2["parents"] for s in combs]
            ancestry[(t2, t1)] = "possible"

    return {"adj_sets": str(adj_sets), "ancestry": str(ancestry)}
