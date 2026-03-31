from collections import defaultdict
from typing import Callable, NamedTuple, Sequence

import numpy as np

from algorithms.mb_by_mb import mb_by_mb_alg, get_locally_valid_parent_sets, grow_shrink
from algorithms.cmb import cmb_alg


class Neighbors(NamedTuple):
    """
    Represents the different types of neighbors for a node in a causal graph.
    """

    parents: set[int]
    children: set[int]
    unoriented: set[int]


def get_neighbors(g: np.ndarray, x: int) -> Neighbors:
    """
    Get the neighbors of a target node in the graph.

    Args:
        g (np.ndarray): The local graph of x.
        x (int): The target node.

    Returns:
        Neighbors: The parents, children, and unoriented neighbors of the target node.
    """
    parents = set(np.where((g[x] == 1) & (g[:, x] == -1))[0])
    children = set(np.where((g[x] == -1) & (g[:, x] == 1))[0])
    unoriented = set(np.where((g[x] == -1) & (g[:, x] == -1))[0])
    return Neighbors(parents, children, unoriented)


def is_explicit_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is an explicit ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        g (np.ndarray): The local graph of x.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.

    Returns:
        bool: True if x is an explicit ancestor of y, False otherwise.
    """
    neighbors = get_neighbors(g, x)
    if y in neighbors.children:
        return True
    elif y in neighbors.parents | neighbors.unoriented:
        return False
    return ci_test(x, y, neighbors.parents | neighbors.unoriented) < alpha


def is_possible_ancestor(
    x: int,
    y: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Check if x is a possible ancestor of y.

    Args:
        x (int): The node to check as ancestor.
        y (int): The node to check as descendant.
        g (np.ndarray): The local graph of x.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if x is a possible ancestor of y, False otherwise.
    """
    neighbors = get_neighbors(g, x)
    if y in neighbors.children | neighbors.unoriented:
        return True
    elif y in neighbors.parents:
        return False
    return ci_test(x, y, neighbors.parents) < alpha


def is_amenable(
    x: int,
    y: int,
    v: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
) -> bool:
    """
    Locally test if an undirected neighbor of the treatment has no undirected paths
    to the outcome that do not go through the treatment.

    Args:
        x (int): The treatment node.
        y (int): The outcome node.
        v (int): An undirected neighbor of the treatment.
        g (np.ndarray): The local graph of v.
        ci_test (Callable): Conditional independence test.
        alpha (float): Significance level.

    Returns:
        bool: True if v does not contradict amenability, otherwise False.
    """
    neighbors = get_neighbors(g, v)
    if y in neighbors.children | neighbors.unoriented | neighbors.parents:  # adjacent
        return False
    return ci_test(v, y, neighbors.parents | {x}) >= alpha


def load(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
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
        targets (Sequence[int]): The target nodes.
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

    t1, t2 = targets
    G[t1] = lcd_alg(data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set)
    G[t2] = lcd_alg(data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set)

    # Determine causal relationship
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1, t2)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2, t1)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "expl_an": None,
            "poss_an": [t1 for t1, _ in adj_sets],
            "step": 1,
        }

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
                "expl_an": treatment,
                "poss_an": [],
                "step": 2,
            }

    # Identify possible descendants of treatment
    pdesc = set()
    for v in set(range(data.shape[1])) - {treatment, outcome}:
        if is_possible_ancestor(treatment, v, G[treatment], ci_test, alpha):
            pdesc.add(v)

    # Obtain optimal adjustment set via forbidden projection
    known_mb = get_neighbors(G[outcome], outcome).parents - pdesc
    oset = grow_shrink(data, outcome, ci_test, alpha, pdesc, known_mb) - {treatment}

    adj_sets[(treatment, outcome)] = [oset]
    return {
        "adj_sets": str(adj_sets),
        "identifiable": True,
        "expl_an": treatment,
        "poss_an": [],
        "step": 4,
    }
