from collections import defaultdict
from itertools import combinations
from typing import Callable

import numpy as np

from algorithms.orient import v_struc_pc, meek


def skeleton_step(
    order: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
) -> np.ndarray:
    """
    Skeleton step at a given order

    Args:
        order (int): The order of CI tests.
        g (np.ndarray): The current skeleton.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        sep_set (dict[frozenset, list]): The separating sets.

    Returns:
        np.ndarray: The updated (in-place) skeleton.
    """
    for x in range(len(g)):
        Neigh_x = np.where(g[x, :] != 0)[0]
        if len(Neigh_x) < order - 1:
            continue
        for y in Neigh_x:
            curr_neigh = np.where(g[x, :] != 0)[0]
            Neigh_x_noy = np.delete(curr_neigh, np.where(curr_neigh == y))
            for S in combinations(Neigh_x_noy, order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    sep_set[frozenset((x, y))].append(S)
                    break
    return g


def pc(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    **kwargs,
) -> dict[str, list[list[int]]]:
    """
    PC algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict[str, list[list[int]]]: The adjacency matrix of the CPDAG.
    """
    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((data.shape[1], data.shape[1]), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    # Skeleton search
    order = 0
    while np.amax(np.sum(skeleton != 0, axis=1)) > order:
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        order += 1

    # Orient v-structures
    pdag = v_struc_pc(skeleton, sep_set)

    # Orient Meek rules
    cpdag = meek(pdag)

    # Convert CPDAG to networkx adjacency matrix
    amat = np.zeros_like(cpdag, dtype=int)
    amat[cpdag == -1] = 1
    return {"amat": amat.tolist()}
