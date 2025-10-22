from collections import defaultdict
from itertools import chain, combinations
from typing import Callable, NamedTuple, Sequence

import numpy as np

from algorithms.mb_by_mb import mb_by_mb_alg


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


def get_locally_valid_parent_sets(g: np.ndarray, x: int) -> Sequence[set[int]]:
    """
    Get all locally valid parent sets for a given node in the graph, following
    the local IDA algorithm (Algorithm 3) in
    Estimating high-dimensional intervention effects from observational data
    by Marloes H. Maathuis, Markus Kalisch and Peter Bühlmann

    Args:
        g (np.ndarray): The local graph of x.
        x (int): The target node.

    Returns:
        Sequence[set[int]]: The locally valid parent sets.
    """
    neighbors = get_neighbors(g, x)
    skeleton = g != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = []
    # for each subset of unoriented neighbors of x
    for new_parents in chain.from_iterable(
        combinations(neighbors.unoriented, r)
        for r in range(len(neighbors.unoriented) + 1)
    ):
        candidate_set = neighbors.parents.union(new_parents)
        # Check if the candidate set is locally valid, i.e., has no NEW v-structure
        # by checking if all NEW parents are neighbours of all current parents
        if np.all(skeleton[new_parents, :][:, list(candidate_set)]):
            valid_sets.append(candidate_set)
    return valid_sets


def load(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    targets: Sequence[int],
    mb_algorithm: str = "grow_shrink",
    ignore: Sequence[int] = [],
    **kwargs,
) -> dict:
    """
    Optimal adjustment set discovery using local causal discovery algorithms.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, Sequence[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (Sequence[int]): The target nodes.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        ignore (Sequence[int]): Nodes to ignore in discovery process.
        **kwargs: Additional keyword arguments are ignored.
    Returns:
        dict: Adjustment sets and a boolean indicating if the causal effect is identifiable.
    """
    MB = dict()
    L = dict()
    G = dict()
    sep_set = defaultdict(list)

    adj_sets = dict()

    t1, t2 = targets
    G[t1] = mb_by_mb_alg(
        data, ci_test, alpha, t1, mb_algorithm, MB, L, sep_set, ignore=ignore
    )
    G[t2] = mb_by_mb_alg(
        data, ci_test, alpha, t2, mb_algorithm, MB, L, sep_set, ignore=ignore
    )

    # Determine causal relationship
    if is_explicit_ancestor(t1, t2, G[t1], ci_test, alpha):
        treatment, outcome = t1, t2
    elif is_explicit_ancestor(t2, t1, G[t2], ci_test, alpha):
        treatment, outcome = t2, t1
    else:
        if is_possible_ancestor(t1, t2, G[t1], ci_test, alpha):
            adj_sets[(t1, t2)] = get_locally_valid_parent_sets(G[t1], t1)
        if is_possible_ancestor(t2, t1, G[t2], ci_test, alpha):
            adj_sets[(t2, t1)] = get_locally_valid_parent_sets(G[t2], t2)
        return {
            "adj_sets": str(adj_sets),
            "identifiable": False,
            "id_tests": ci_test.get_tests_per_order().tolist(),
        }

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
    oset = set()
    for med in meds | {outcome}:
        oset |= get_neighbors(G[med], med).parents
    oset -= meds | {treatment}

    adj_sets[(treatment, outcome)] = [oset]
    return {"adj_sets": str(adj_sets), "identifiable": True, "id_tests": id_tests}
