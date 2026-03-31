from collections import defaultdict, deque
from itertools import chain, combinations
from typing import Callable, Sequence

import numpy as np

from algorithms.orient import v_struc_pc
from algorithms.pc import skeleton_step
from algorithms.s2tmb import s2tmb


def get_locally_valid_parent_sets(
    g: np.ndarray, treatment: int, outcome: int
) -> list[set[int]]:
    """
    Get all locally valid parent sets for a given node in the graph, following
    the local IDA algorithm (Algorithm 3) in
    Estimating high-dimensional intervention effects from observational data
    by Marloes H. Maathuis, Markus Kalisch and Peter Bühlmann

    Args:
        g (np.ndarray): The graph.
        treatment (int): The treatment node.
        outcome (int): The outcome node.

    Returns:
        list[set[int]]: The locally valid parent sets.
    """
    parents = set(np.where((g[treatment] == 1) & (g[:, treatment] == -1))[0])
    siblings = set(np.where((g[treatment] == -1) & (g[:, treatment] == -1))[0])
    siblings = siblings - {outcome}
    skeleton = g != 0
    np.fill_diagonal(skeleton, True)
    valid_sets = [parents]
    for l in range(1, len(siblings) + 1):
        for sib in combinations(siblings, l):
            candidate_set = set(sib) | parents
            if np.all(skeleton[list(sib), :][:, list(candidate_set)]):
                valid_sets.append(candidate_set)
    return valid_sets


def grow_shrink(
    data: np.ndarray,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    ignore: set | None = None,
    known_mb: set | None = None,
) -> set:
    """
    Find the Markov blanket of a target node in a graph using the Grow-Shrink algorithm.

    Args:
        data (np.ndarray): The data matrix.
        target (int): Target node for which to find the Markov blanket.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
        ignore (set | None): Nodes to ignore.
        known_mb (set | None): Nodes known to be in the Markov blanket.
    Returns:
        set: The Markov blanket of the target node.
    """
    if ignore is None:
        ignore = set()
    if known_mb is None:
        known_mb = set()
    nodes = set(range(data.shape[1])) - {target} - ignore
    mb = known_mb.copy()
    # Forward pass
    cont = True
    while cont:
        cont = False
        mb_copy = set(mb)
        nodes_to_check = nodes - mb_copy
        for n in nodes_to_check:
            if ci_test(n, target, mb - {n}) < alpha:
                mb.add(n)
                cont = True

    # Backward pass
    mb_copy = set(mb)
    for n in mb_copy - known_mb:
        if ci_test(n, target, mb - {n}) >= alpha:
            mb.remove(n)

    return mb


def total_conditioning(
    data: np.ndarray,
    target: int,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    ignore: set | None = None,
    known_mb: set | None = None,
) -> set:
    """
    Find the Markov blanket of a target node in a graph using the Total Conditioning algorithm
    from Using markov blankets for causal structure learning by
    Jean-Philippe Pellet and André Elisseeff.

    Args:
        data (np.ndarray): The data matrix.
        target (int): Target node for which to find the Markov blanket.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
        ignore (set | None): Nodes to ignore.
        known_mb (set | None): Nodes known to be in the Markov blanket.
    Returns:
        set: The Markov blanket of the target node.
    """
    if ignore is None:
        ignore = set()
    if known_mb is None:
        known_mb = set()
    nodes = set(range(data.shape[1])) - {target} - ignore
    mb = known_mb.copy()
    for n in nodes - known_mb:
        if ci_test(n, target, nodes - {target, n}) < alpha:
            mb.add(n)
    return mb


def learn_local_structure(
    data: np.ndarray,
    observed: set,
    sep_set: dict[frozenset, list],
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
) -> np.ndarray:
    """
    Learn the local structure over a set of observed nodes using the PC algorithm.

    Args:
        data (np.ndarray): The data matrix.
        observed (set): The observed nodes.
        sep_set (dict[frozenset, list]): The separating sets.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level for the CI test.
    Returns:
        np.ndarray: The adjacency matrix of the learned local structure.
    """
    # Initialization
    nodes = set(range(data.shape[1]))
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


def reach_with_undirected(g: np.ndarray, target: int) -> np.ndarray:
    """
    Find all nodes that can reach the target node with an undirected path with BFS.
    Args:
        g (np.ndarray): The partially directed graph.
        target (int): The target node.
    Returns:
        np.ndarray: An array of nodes that can reach the target node with an undirected path
    """
    undir = (g == -1) & (g.T == -1)

    # BFS from target
    visited = np.zeros(g.shape[0], dtype=bool)
    visited[target] = True
    queue = deque([target])

    while queue:
        node = queue.popleft()
        # Get all undirected neighbors of current node
        neighbors = np.where(undir[node])[0]
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    visited[target] = False
    return set(np.where(visited)[0])


def mb_by_mb_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    mb_algorithm: str = "grow_shrink",
    mb: dict[int, set] | None = None,
    L: dict[int, np.ndarray] | None = None,
    sep_set: dict[frozenset, list] | None = None,
    ignore: set = set(),
) -> np.ndarray:
    """
    MB-by-MB algorithm for learning the local network around a target node.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        target (int): The target node for which to learn the local network.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        mb (dict[int, set]): Pre-computed Markov blankets of nodes.
        L (dict[int, np.ndarray]): Pre-computed local structures of nodes.
        sep_set (dict[frozenset, list]): Pre-computed separating sets.
        ignore (set): Nodes to ignore.
    Returns:
        np.ndarray: The learned local network around the target node.
    """
    # Input
    if mb_algorithm == "grow_shrink":
        mb_algorithm = grow_shrink
    elif mb_algorithm == "total_conditioning":
        mb_algorithm = total_conditioning
    elif mb_algorithm == "s2tmb":
        mb_algorithm = s2tmb
    else:
        raise ValueError(f"Unknown mb_algorithm: {mb_algorithm}")

    if mb is None:
        mb = dict()
    if L is None:
        L = dict()
    if sep_set is None:
        sep_set = defaultdict(list)

    # 1. Initialization
    # Nodes whose MBs have been found
    done_list = set()
    # Nodes whose MBs will be found
    wait_list = deque([target])
    # The constructed local network around target
    # with -1 for edge tail, 1 for edge head, and 0 for no edge
    g = np.zeros((data.shape[1], data.shape[1]), dtype=int)

    # 2. Repeat
    while len(wait_list) > 0:  # 7. Until wait_list is empty
        # Take a node x from the head of wait_list
        x = wait_list.popleft()
        # Find mb[x]
        if x not in mb:
            mb[x] = mb_algorithm(data, x, ci_test, alpha, ignore)
        mb_x = mb[x].union({x})
        # Add [mb[x] \ done_nodes \ wait_list] to the tail of wait_list
        wait_list.extend(mb[x] - done_list - set(wait_list))
        # Add x to done_nodes
        done_list.add(x)

        # 3. Learn the local structure L[x] over mb+(x)
        # If L[x] is already learned, skip
        if x in L:
            pass
        # If mb+(x) is a subset of mb+(n) for some n in done_list
        elif (
            n := next((n for n in done_list if n != x and mb_x <= mb[n] | {n}), None)
        ) is not None:
            # Set L[x] equal to the substructure of L[n] over mb+(x)
            L[x] = copy_local_structure(L[n], mb_x)
        # Else If mb(x) is a subset of done_list
        elif mb[x] < done_list:
            # Set L[x] equal to the substructure of g over mb+(x)
            L[x] = copy_local_structure(g, mb_x)
        else:
            # learn L[x] from observed data of mb[x]+
            L[x] = learn_local_structure(data, mb_x, sep_set, ci_test, alpha)

        # 4. Put the edges connected to x and the v-structures containing x in L[x] to g
        g = update_graph(g, x, L[x])
        # 5. Orient undirected edges in G
        g = meek(g, sep_set)

        # 6. Remove all nodes from wait_list whose paths to target in g are blocked by directed edges.
        connected = reach_with_undirected(g, target)
        wait_list = deque(v for v in wait_list if v in connected)

    # Output: the local network g around the target
    return g


def mb_by_mb(
    data: np.ndarray,
    ci_test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    treatment: int,
    outcome: int,
    mb_algorithm: str = "grow_shrink",
    **kwargs,
) -> dict:
    """
    MB-by-MB algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        treatment (int): The treatment node.
        outcome (int): The outcome node.
        mb_algorithm (str): The algorithm to use for finding the Markov blanket.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Locally valid parent adjustment sets of treatment.
    """
    G = mb_by_mb_alg(data, ci_test, alpha, treatment, mb_algorithm)
    local_sets = get_locally_valid_parent_sets(G, treatment, outcome)
    adj_sets = {(treatment, outcome): local_sets}
    return {"adj_sets": str(adj_sets)}
