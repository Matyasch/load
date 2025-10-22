from collections import defaultdict
from typing import Callable

import numpy as np

from algorithms.orient import v_struc_pc, v_struc_rfci, meek
from algorithms.pc import skeleton_step


def get_poss_anc(g: np.ndarray, targets: list[int]) -> list[int]:
    """
    Get possible ancestors of any target.

    Args:
        g (np.ndarray): The causal graph.
        targets (list[int]): The target nodes.

    Returns:
        list[int]: Possible ancestors of any target.
    """
    g = -g.copy()
    g[g == -1] = 0
    np.fill_diagonal(g, 1)
    g = g.astype(bool)
    # Reachability matrix
    reach = np.linalg.matrix_power(g, g.shape[0] - 1)
    # Nodes that reach at least one target
    poss_anc = np.any(reach[:, targets], axis=1)
    return np.where(poss_anc)[0]


def snap(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    targets: list[int],
    max_order: int = -1,
    **kwargs,
) -> list[int] | dict:
    """
    SNAP algorithm to estimate the CPDAG over the possibly ancestral set of the targets, up to a maximum order of CI tests.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        max_order (int): The maximum order of CI tests for SNAP(k). If negative, run SNAP(infinity) until completion (default: -1).
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Adjacency matrix of pruned CPDAG and possible ancestors of targets.
    """
    num_nodes = data.shape[1]
    if max_order < 0:  # Run SNAP(infinity) until completion
        max_order = num_nodes - 1

    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    all_nodes = poss_anc = np.arange(num_nodes)
    for order in range(max_order + 1):
        if np.amax(np.sum(skeleton != 0, axis=1)) <= order:
            break
        # Perform skeleton step
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        # Orient v-structures
        if order < 2:
            pdag = v_struc_pc(skeleton, sep_set, bidirected=True)
        else:
            pdag, skeleton = v_struc_rfci(skeleton, ci_test, alpha, sep_set)
        # Get possibly ancestral set containing targets
        poss_anc = get_poss_anc(pdag, targets)
        # Prune non-ancestors
        non_anc = np.setdiff1d(all_nodes, poss_anc)
        skeleton[non_anc, :] = skeleton[:, non_anc] = 0

    # If ran until completion
    if max_order == num_nodes - 1:
        # Orient v-structures without bidirected edges
        pdag = v_struc_pc(skeleton, sep_set)
        # Orient Meek rules
        res = meek(pdag)
        # Prune non-ancestors one last time
        poss_anc = get_poss_anc(res, targets)
        non_anc = np.setdiff1d(all_nodes, poss_anc)
        res[non_anc, :] = res[:, non_anc] = 0
    else:
        res = pdag

    # Convert to networkx format
    amat = np.zeros_like(res, dtype=int)
    amat[res == -1] = 1
    return {"amat": amat.tolist(), "poss_anc": poss_anc.tolist()}
