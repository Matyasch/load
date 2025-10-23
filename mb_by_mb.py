from collections import defaultdict
from itertools import combinations
from typing import Callable

import numpy as np


# Grow-Shrink algorithm for Markov blanket discovery
def grow_shrink(
    nodes: set,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> set:
    """
    Find the Markov blanket of a target node in a graph using the Grow-Shrink algorithm.
    Adapted from https://github.com/acmi-lab/local-causal-discovery.

    Args:
        nodes (set): All nodes.
        target (int): Target node for which to find the Markov blanket.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
    Returns:
        set: The Markov blanket of the target node.
    """
    mb = set()
    # Forward pass
    cont = True
    while cont:
        cont = False
        mb_copy = set(mb)
        nodes_to_check = nodes - {target} - mb_copy
        for n in nodes_to_check:
            if ci_test(n, target, mb - {n}) < alpha:
                mb.add(n)
                cont = True

    # Backward pass
    mb_copy = set(mb)
    for n in mb_copy:
        if ci_test(n, target, mb - {n}) >= alpha:
            mb.remove(n)

    return mb


# Learn local structure using the PC algorithm
def find_unshielded_triples(g: np.ndarray) -> np.ndarray:
    """
    Find all unshielded triples in the skeleton in a vectorized way.

    Args:
        g (np.ndarray): The skeleton.

    Returns:
        np.ndarray: The unshielded triples.
    """
    i, j = np.where(g != 0)  # i - j
    k = np.where((g[j] != 0) & (g[i] == 0))  # j - k -/- i
    # Extract actual indices
    i, j, k = i[k[0]], j[k[0]], k[1]
    # Ensure no duplicates
    mask = i < k
    return np.column_stack((i[mask], j[mask], k[mask]))


def v_struc_pc(
    skeleton: np.ndarray, sep_set: dict[frozenset, list], bidirected: bool = False
) -> np.ndarray:
    """
    Orient v-structures in the skeleton as in the PC algorithm

    Args:
        skeleton (np.ndarray): The skeleton to orient.
        sep_set (dict[frozenset, list]): The separating sets.
        bidirected (bool): Whether to orient conflicts as bidirected, or overwrite.

    Returns:
        np.ndarray: The PDAG.
    """
    pdag = skeleton.copy()
    for x, y, z in find_unshielded_triples(skeleton):
        if all(y not in S for S in sep_set[frozenset((x, z))]):
            pdag[y, x] = pdag[y, z] = 1  # Orient arrowhead
            if not bidirected:
                pdag[x, y] = pdag[z, y] = -1  # Orient tail
    return pdag


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


def learn_local_structure(
    nodes: set,
    observed: set,
    sep_set: dict[frozenset, list],
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> np.ndarray:
    """
    Learn the local structure over a set of observed nodes using the PC algorithm.

    Args:
        nodes (set): All nodes.
        observed (set): The observed nodes.
        sep_set (dict[frozenset, list]): The separating sets.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
    Returns:
        np.ndarray: The adjacency matrix of the learned local structure.
    """
    # Initialization
    g = np.full((len(nodes), len(nodes)), -1, dtype=int)
    np.fill_diagonal(g, 0)
    ignore = list(nodes.difference(observed))
    g[ignore, :] = g[:, ignore] = 0

    # Skeleton search
    order = 0
    while np.amax(np.sum(g != 0, axis=1)) > order:
        g = skeleton_step(order, g, ci_test, alpha, sep_set)
        order += 1
    # Orient v-structures
    g = v_struc_pc(g, sep_set, bidirected=True)

    return g


def copy_local_structure(g: np.ndarray, observed: set) -> np.ndarray:
    """
    Create a copy of the local structure over the observed nodes.

    Args:
        g (np.ndarray): The input graph.
        observed (set): The observed nodes.

    Returns:
        np.ndarray: A copy of the local structure.
    """
    local = np.zeros_like(g, dtype=int)
    observed = list(observed)
    local[np.ix_(observed, observed)] = g[np.ix_(observed, observed)]
    return local


# Update global graph with local structure
def update_graph(g: np.ndarray, target: int, local: np.ndarray) -> np.ndarray:
    """
    Update the graph with the local structure around the target node.

    Args:
        g (np.ndarray): The global graph.
        target (int): The target node.
        local (np.ndarray): Local structure for the target node.
    Returns:
        np.ndarray: The updated global graph.
    """
    # Put the edges connected to target in local to g
    g[target, :] = local[target, :]
    g[:, target] = local[:, target]
    # Put v-structures containing target as a parent in local to g.
    coll = np.where(local.T[target] == 1)[0]  # target *-> coll
    spouse = np.where((local[coll] == 1) & (local[target] == 0))
    coll, spouse = coll[spouse[0]], spouse[1]
    mask = spouse != target
    coll, spouse = coll[mask], spouse[mask]
    g[target, coll] = local[target, coll]
    g[coll, target] = local[coll, target]
    g[spouse, coll] = local[spouse, coll]
    g[coll, spouse] = local[coll, spouse]

    return g


def meek(g: np.ndarray, sep_set: dict[frozenset, list]) -> np.ndarray:
    """
    Orient the undirected edges in g by a revision of Meek's approach,
    as described in the MB-by-MB algorithm.

    Args:
        g (np.ndarray): The graph to orient.
        sep_set (dict[frozenset, list]): The separating sets.
    Returns:
        np.ndarray: The oriented graph.
    """

    def rule1(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        For (a -> b - c) in g,
        if g.sepset[a, c] exists and b in g.sepset[a, c], orient b -> c.

        Args:
            g (np.ndarray): The graph to orient.
        Returns:
            bool: Whether the graph was changed.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(((g[b] == -1) & (g.T[b] == -1)) & (g[a] == 0))  # b - c  # a -/- c
        # Extract actual indices
        a, b, c = a[c[0]], b[c[0]], c[1]
        # Orient as b -> c
        changed = False
        for a_, b_, c_ in zip(a, b, c):
            if frozenset((a_, c_)) in sep_set and all(
                b_ in S for S in sep_set[frozenset((a_, c_))]
            ):
                g[c_, b_] = 1
                changed = True

        return changed

    def rule2(g: np.ndarray) -> bool:
        """
        For (a -> b -> c - a) in g, orient a -> c.
        Args:
            g (np.ndarray): The graph to orient.
        Returns:
            bool: Whether the graph was changed.
        """
        a, b = np.where((g == -1) & (g.T == 1))  # a -> b
        c = np.where(
            ((g[b] == -1) & (g.T[b] == 1))  # b -> c
            & ((g[a] == -1) & (g.T[a] == -1))  # a - c
        )
        # Extract actual indices
        a, b, c = a[c[0]], b[c[0]], c[1]
        # Orient as a -> c
        g[c, a] = 1
        # return whether graph changed
        return len(a) > 0

    def rule3(g: np.ndarray, sep_set: dict[frozenset, list]) -> bool:
        """
        For a - b, a - c -> b and a - d -> b in g,
        if g.sepset[c, d] exists and a in g.sepset[c, d], orient a -> b.
        """
        a, c = np.where((g == -1) & (g.T == -1))  # a - c
        d = np.where(((g[a] == -1) & (g.T[a] == -1)) & (g[c] == 0))  # a - d -/- c
        # Extract actual indices
        a, c, d = a[d[0]], c[d[0]], d[1]
        mask = c < d
        a, c, d = a[mask], c[mask], d[mask]
        b = np.where(
            ((g[a] == -1) & (g.T[a] == -1))  # a - b
            & ((g[c] == -1) & (g.T[c] == 1))  # c -> b
            & ((g[d] == -1) & (g.T[d] == 1))  # d -> b
        )
        # Extract actual indices
        a, c, d, b = a[b[0]], c[b[0]], d[b[0]], b[1]
        # Orient as i -> l
        changed = False
        for a_, b_, c_, d_ in zip(a, b, c, d):
            if frozenset((c_, d_)) in sep_set and all(
                a_ in S for S in sep_set[frozenset((c_, d_))]
            ):
                g[b_, a_] = 1
                changed = True
        return changed

    g = g.copy()
    while rule1(g, sep_set) or rule2(g) or rule3(g, sep_set):
        continue
    return g


# Test stopping criterion
def reach_with_undirected(g: np.ndarray, target: int) -> np.ndarray:
    """
    Find all nodes that can reach the target node with an undirected path.
    Args:
        g (np.ndarray): The partially directed graph.
        target (int): The target node.
    Returns:
        np.ndarray: An array of nodes that can reach the target node with an undirected path
    """
    undir = (g == -1) & (g.T == -1)
    np.fill_diagonal(undir, 1)
    reach = np.linalg.matrix_power(undir, undir.shape[0] - 1)
    mask = reach[target] != 0
    mask[target] = False
    nodes = np.where(mask)[0]
    return nodes


# Main MB-by-MB algorithm
def mb_by_mb(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
) -> np.ndarray:
    """
    MB-by-MB algorithm for learning the local network around a target node.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        target (int): The target node for which to learn the local network.
        mb (dict[int, set]): Pre-computed Markov blankets of nodes.
        L (dict[int, np.ndarray]): Pre-computed local structures of nodes.
        sep_set (dict[frozenset, list]): Pre-computed separating sets.
    Returns:
        np.ndarray: The learned local network around the target node.
    """
    mb = mb or dict()
    L = L or dict()
    sep_set = sep_set or defaultdict(list)
    V = set(range(data.shape[1]))

    # 1. Initialization
    # Nodes whose MBs have been found
    done_list = set()
    # Nodes whose MBs will be found
    wait_list = [target]
    # The constructed local network around target
    # with -1 for edge tail, 1 for edge head, and 0 for no edge
    g = np.zeros((len(V), len(V)), dtype=int)

    # 2. Repeat
    while len(wait_list) > 0:  # 7. Until wait_list is empty
        # Take a node x from the head of wait_list
        x = wait_list.pop(0)
        # Find mb[x]
        if x not in mb:
            mb[x] = grow_shrink(V, x, ci_test, alpha)
        mb_x = mb[x].union({x})
        # Add [mb[x] \ done_list \ wait_list] to the tail of wait_list
        wait_list.extend([v for v in mb[x] if v not in list(done_list) + wait_list])
        # Add x to done_list
        done_list.add(x)

        # 3. Learn the local structure L[x] over mb+(x)
        # If L[x] is already learned, skip
        if x in L:
            pass
        # If mb+(x) is a subset of mb+(n) for some n in done_list
        elif (
            n := next((n for n in done_list if n != x and mb_x < mb[n] | {n}), None)
        ) is not None:
            # Set L[x] equal to the substructure of L[n] over mb+(x)
            L[x] = copy_local_structure(L[n], mb_x)
        # Else If mb(x) is a subset of done_list
        elif mb[x] < done_list:
            # Set L[x] equal to the substructure of g over mb+(x)
            L[x] = copy_local_structure(g, mb_x)
        else:
            # learn L[x] from observed data of mb[x]+
            L[x] = learn_local_structure(V, mb_x, sep_set, ci_test, alpha)

        # 4. Put the edges connected to x and the v-structures containing x in L[x] to g
        g = update_graph(g, x, L[x])
        # 5. Orient undirected edges in G
        g = meek(g, sep_set)

        # 6. Remove all nodes from wait_list whose paths to target in g are blocked by directed edges.
        connected = reach_with_undirected(g, target)
        wait_list = [w for w in wait_list if w in connected]

    # Output: the local network g around the target
    return g
