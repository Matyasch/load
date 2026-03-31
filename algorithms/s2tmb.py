"""
Most of the code in this file is copied from the pyCausalFS package
(https://github.com/wt-hu/pyCausalFS) by Wentao Hu et al., 2020.
"""

import numpy as np
import networkx as nx


def topsort(edge_dict, root=None):
    """
    List of nodes in topological sort order from edge dict
    where key = rv and value = list of rv's children
    """
    queue = []
    if root is not None:
        queue = [root]
    else:
        for rv in edge_dict.keys():
            prior = True
            for p in edge_dict.keys():
                if rv in edge_dict[p]:
                    prior = False
            if prior == True:
                queue.append(rv)

    visited = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            for nbr in edge_dict[vertex]:
                queue.append(nbr)
            # queue.extend(edge_dict[vertex]) # add all vertex's children
    return visited


class BayesNet(object):
    """
    **************
    BayesNet Class
    **************

    Overarching class for Discrete Bayesian Networks.


    Design Specs
    ------------

    - Bayesian Network -

        - F -
            key:
                - rv -
            values:
                - children -
                - parents -
                - values -
                - cpt -

        - V -
            - list of rvs -

        - E -
            key:
                - rv -
            values:
                - list of rv's children -
    Notes
    -----
    - Edges can be inferred from Factorization, but Vertex values must be specified.
    """

    __author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

    def __init__(self, E=None, value_dict=None):
        """
        Initialize the BayesNet class.

        Arguments
        ---------
        *V* : a list of strings - vertices in topsort order
        *E* : a dict, where key = vertex, val = list of its children
        *F* : a dict,
            where key = rv,
            val = another dict with
                keys =
                    'parents',
                    'values',
                    'cpt'

        *V* : a dict

        Notes
        -----

        """
        if E is not None:
            # assert (value_dict is not None), 'Must set values if E is set.'
            self.set_structure(E, value_dict)
        else:
            self.V = []
            self.E = {}
            self.F = {}

    # def __eq__(self, y):
    #     """
    #     Tests whether two Bayesian Networks are
    #     equivalent - i.e. they contain the same
    #     node/edge structure, and equality of
    #     conditional probabilities.
    #     """
    #     return are_class_equivalent(self, y)

    def __hash__(self):
        """
        Allows BayesNet objects to be used
        as keys in a dictionary (i.e. hashable)
        """
        return hash((str(self.V), str(self.E)))

    def add_node(self, rv, cpt=[], parents=[], values=[]):
        self.V.append(rv)
        self.F[rv] = {"cpt": cpt, "parents": parents, "values": values}

    def add_edge(self, u, v):
        if not self.has_node(u):
            self.add_node(u)
        if not self.has_node(v):
            self.add_node(v)
        if self.has_edge(u, v):
            print("Edge already exists")
        else:
            self.E[u].append(v)
            self.F[v]["parents"].append(u)
        # self.V = topsort(self.E)
        # HOW DO I RECALCULATE CPT?

    def remove_edge(self, u, v):
        self.E[u].remove(v)
        self.F[v]["parents"].remove(u)

    def reverse_arc(self, u, v):
        if self.has_edge(u, v):
            self.E[u].remove(v)
            self.E[v].append(u)

    def set_data(self, rv, data):
        assert isinstance(data, dict), "data must be dictionary"
        self.F[rv] = data

    def set_cpt(self, rv, cpt):
        self.F[rv]["cpt"] = cpt

    def set_parents(self, rv, parents):
        self.F[rv]["parents"] = parents

    def set_values(self, rv, values):
        self.F[rv]["values"] = values

    def nodes(self):
        for v in self.V:
            yield v

    def node_idx(self, rv):
        try:
            return self.V.index(rv)
        except ValueError:
            return -1

    def has_node(self, rv):
        return rv in self.V

    def has_edge(self, u, v):
        return v in self.E[u]

    def num_edges(self):
        num = 0
        for u in self.nodes():
            num += len(self.E[u])
        return num

    def num_params(self):
        num = 0
        for u in self.nodes():
            num += len(self.F[u]["cpt"])
        return num

    def scope_size(self, rv):
        return len(self.F[rv]["parents"]) + 1

    def num_nodes(self):
        return len(self.V)

    def cpt(self, rv):
        return self.F[rv]["cpt"]

    def card(self, rv):
        return len(self.F[rv]["values"])

    def scope(self, rv):
        scope = [rv]
        scope.extend(self.F[rv]["parents"])
        return scope

    def parents(self, rv):
        return self.F[rv]["parents"]

    def children(self, rv):
        return self.E[rv]

    def degree(self, rv):
        return len(self.parents(rv)) + len(self.children(rv))

    def values(self, rv):
        return self.F[rv]["values"]

    def value_idx(self, rv, val):
        try:
            return self.F[rv]["values"].index(val)
        except ValueError:
            print("Value Index Error")
            return -1

    def stride(self, rv, n):
        if n == rv:
            return 1
        else:
            card_list = [self.card(rv)]
            card_list.extend([self.card(p) for p in self.parents(rv)])
            n_idx = self.parents(rv).index(n) + 1
            return int(np.prod(card_list[0:n_idx]))

    def flat_cpt(self, by_var=False, by_parents=False):
        """
        Return all cpt values in the BN as a flattened
        numpy array ordered by bn.nodes() - i.e. topsort
        """
        if by_var:
            cpt = np.array([sum(self.cpt(rv)) for rv in self.nodes()])
        elif by_parents:
            cpt = np.array(
                [
                    sum(self.cpt(rv)[i : (i + self.card(rv))])
                    for rv in self.nodes()
                    for i in range(len(self.cpt(rv)) / self.card(rv))
                ]
            )
        else:
            cpt = np.array([val for rv in self.nodes() for val in self.cpt(rv)])
        return cpt

    def cpt_indices(self, target, val_dict):
        """
        Get the index of the CPT which corresponds
        to a dictionary of rv=val sets. This can be
        used for parameter learning to increment the
        appropriate cpt frequency value based on
        observations in the data.

        There is definitely a fast way to do this.
            -- check if (idx - rv_stride*value_idx) % (rv_card*rv_stride) == 0

        Arguments
        ---------
        *target* : a string
            Main RV

        *val_dict* : a dictionary, where
            key=rv,val=rv value

        """
        stride = dict([(n, self.stride(target, n)) for n in self.scope(target)])
        # if len(val_dict)==len(self.parents(target)):
        #    idx = sum([self.value_idx(rv,val)*stride[rv] \
        #            for rv,val in val_dict.items()])
        # else:
        card = dict([(n, self.card(n)) for n in self.scope(target)])
        idx = set(range(len(self.cpt(target))))
        for rv, val in val_dict.items():
            val_idx = self.value_idx(rv, val)
            rv_idx = []
            s_idx = val_idx * stride[rv]
            while s_idx < len(self.cpt(target)):
                rv_idx.extend(range(s_idx, (s_idx + stride[rv])))
                s_idx += stride[rv] * card[rv]
            idx = idx.intersection(set(rv_idx))

        return list(idx)

    def cpt_str_idx(self, rv, idx):
        """
        Return string representation of RV=VAL and
        Parents=Val for the given idx of the given rv's cpt.
        """
        rv_val = self.values(rv)[idx % self.card(rv)]
        s = str(rv) + "=" + str(rv_val) + "|"
        _idx = 1
        for parent in self.parents(rv):
            for val in self.values(parent):
                if idx in self.cpt_indices(rv, {rv: rv_val, parent: val}):
                    s += str(parent) + "=" + str(val)
                    if _idx < len(self.parents(rv)):
                        s += ","
                    _idx += 1
        return s

    def set_structure(self, edge_dict, value_dict=None):
        """
        Set the structure of a BayesNet object. This
        function is mostly used to instantiate a BN
        skeleton after structure learning algorithms.

        See "structure_learn" folder & algorithms

        Arguments
        ---------
        *edge_dict* : a dictionary,
            where key = rv,
            value = list of rv's children
            NOTE: THIS MUST BE DIRECTED ALREADY!

        *value_dict* : a dictionary,
            where key = rv,
            value = list of rv's possible values

        Returns
        -------
        None

        Effects
        -------
        - sets self.V in topsort order from edge_dict
        - sets self.E
        - creates self.F structure and sets the parents

        Notes
        -----

        """

        self.V = topsort(edge_dict)
        self.E = edge_dict
        self.F = dict([(rv, {}) for rv in self.nodes()])
        for rv in self.nodes():
            self.F[rv] = {
                "parents": [p for p in self.nodes() if rv in self.children(p)],
                "cpt": [],
                "values": [],
            }
            if value_dict is not None:
                self.F[rv]["values"] = value_dict[rv]

    def moralized_edges(self):
        """
        Moralized graph is the original graph PLUS
        an edge between every set of common effect
        structures -
            i.e. all parents of a node are connected.

        This function has be validated.

        Returns
        -------
        *e* : a python list of parent-child tuples.

        """
        e = set()
        for u in self.nodes():
            for p1 in self.parents(u):
                e.add((p1, u))
                for p2 in self.parents(u):
                    if p1 != p2 and (p2, p1) not in e:
                        e.add((p1, p2))
        return list(e)


def mle_estimator(bn: BayesNet, data, nodes=None, counts=False):
    if nodes is None:
        nodes = list(bn.nodes())
    else:
        if not isinstance(nodes, list):
            nodes = list(nodes)

    F = dict([(rv, {}) for rv in nodes])
    for i, n in enumerate(nodes):
        F[n]["values"] = list(np.unique(data[:, i]))
        bn.F[n]["values"] = list(np.unique(data[:, i]))

    obs_dict = dict([(rv, []) for rv in nodes])
    # set empty conditional probability table for each RV
    for rv in nodes:
        # get number of values in the CPT = product of scope vars' cardinalities
        p_idx = int(np.prod([bn.card(p) for p in bn.parents(rv)]) * bn.card(rv))
        F[rv]["cpt"] = [0] * p_idx
        bn.F[rv]["cpt"] = [0] * p_idx

    # loop through each row of data
    for row in data:
        # store the observation of each variable in the row
        for rv in nodes:
            obs_dict[rv] = row[rv]

        # obs_dict = dict([(rv,row[rv]) for rv in nodes])
        # loop through each RV and increment its observed parent-self value
        for rv in nodes:
            rv_dict = {n: obs_dict[n] for n in obs_dict if n in bn.scope(rv)}
            offset = bn.cpt_indices(target=rv, val_dict=rv_dict)[0]
            F[rv]["cpt"][offset] += 1

    if counts:
        return F
    else:
        for rv in nodes:
            F[rv]["parents"] = [var for var in nodes if rv in bn.E[var]]
            for i in range(0, len(F[rv]["cpt"]), bn.card(rv)):
                temp_sum = float(np.sum(F[rv]["cpt"][i : (i + bn.card(rv))]))
                for j in range(bn.card(rv)):
                    F[rv]["cpt"][i + j] /= temp_sum + 1e-7
                    F[rv]["cpt"][i + j] = round(F[rv]["cpt"][i + j], 5)
        bn.F = F


def BIC(bn: BayesNet, nrow):
    """
    Bayesian Information Criterion.

    BIC = LL - f(N)*|B|, where f(N) = log(N)/2

    """
    log_score = np.sum(np.log(nrow * (bn.flat_cpt() + 1e-7)))
    penalty = 0.5 * bn.num_params() * np.log(max(bn.num_edges(), 1))
    return log_score - penalty


def would_cause_cycle(e, u, v, reverse=False):
    """
    Test if adding the edge u -> v to the BayesNet
    object would create a DIRECTED (i.e. illegal) cycle.
    """
    G = nx.DiGraph(e)
    if reverse:
        G.remove_edge(v, u)
    G.add_edge(u, v)
    try:
        nx.find_cycle(G, source=u)
        return True
    except:
        return False


def unique_bins(data):
    """
    Get the unique values for each column in a dataset.
    """
    bins = np.empty(len(data.T), dtype=np.int32)
    i = 0
    for col in data.T:
        bins[i] = len(np.unique(col))
        i += 1
    return bins


def mutual_information(data, conditional=False):
    # bins = np.amax(data, axis=0)+1 # read levels for each variable
    bins = unique_bins(data)
    if len(bins) == 1:
        hist, _ = np.histogramdd(data, bins=(bins))  # frequency counts
        Px = hist / hist.sum()
        MI = -1 * np.sum(Px * np.log(Px))
        return round(MI, 4)

    if len(bins) == 2:
        hist, _ = np.histogramdd(data, bins=bins[0:2])  # frequency counts

        Pxy = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis=1)  # P(X,Z)
        Py = np.sum(Pxy, axis=0)  # P(Y,Z)

        PxPy = np.outer(Px, Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI, 4)
    elif len(bins) > 2 and conditional == True:
        # CHECK FOR > 3 COLUMNS -> concatenate Z into one column
        if len(bins) > 3:
            data = data.astype("str")
            ncols = len(bins)
            for i in range(len(data)):
                data[i, 2] = "".join(data[i, 2:ncols])
            data = data.astype("int")[:, 0:3]

        bins = np.amax(data, axis=0)
        hist, _ = np.histogramdd(data, bins=bins)  # frequency counts

        Pxyz = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Pz = np.sum(Pxyz, axis=(0, 1))  # P(Z)
        Pxz = np.sum(Pxyz, axis=1)  # P(X,Z)
        Pyz = np.sum(Pxyz, axis=0)  # P(Y,Z)

        Pxy_z = Pxyz / (Pz + 1e-7)  # P(X,Y | Z) = P(X,Y,Z) / P(Z)
        Px_z = Pxz / (Pz + 1e-7)  # P(X | Z) = P(X,Z) / P(Z)
        Py_z = Pyz / (Pz + 1e-7)  # P(Y | Z) = P(Y,Z) / P(Z)

        Px_y_z = np.empty((Pxy_z.shape))  # P(X|Z)P(Y|Z)
        for i in range(bins[0]):
            for j in range(bins[1]):
                for k in range(bins[2]):
                    Px_y_z[i][j][k] = Px_z[i][k] * Py_z[j][k]
        Pxyz += 1e-7
        Pxy_z += 1e-7
        Px_y_z += 1e-7
        MI = np.sum(Pxyz * np.log(Pxy_z / (Px_y_z)))

        return round(MI, 4)
    elif len(bins) > 2 and conditional == False:
        data = data.astype("str")
        ncols = len(bins)
        for i in range(len(data)):
            data[i, 1] = "".join(data[i, 1:ncols])
        data = data.astype("int")[:, 0:2]

        hist, _ = np.histogramdd(data, bins=bins[0:2])  # frequency counts

        Pxy = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis=1)  # P(X,Z)
        Py = np.sum(Pxy, axis=0)  # P(Y,Z)

        PxPy = np.outer(Px, Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI, 4)


def hc(data, metric="BIC", max_iter=100, debug=False, restriction=None):
    nrow = data.shape[0]
    ncol = data.shape[1]

    names = range(ncol)

    # INITIALIZE NETWORK W/ NO EDGES
    # maintain children and parents dict for fast lookups
    c_dict = dict([(n, []) for n in names])
    p_dict = dict([(n, []) for n in names])

    # COMPUTE INITIAL LIKELIHOOD SCORE
    bn = BayesNet(c_dict)
    mle_estimator(bn, data)

    # CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
    # ED = EmpiricalDistribution(data,names)

    _iter = 0
    improvement = True

    while improvement:
        improvement = False
        max_delta = 0

        if debug:
            print("ITERATION: ", _iter)

        ### TEST ARC ADDITIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if (
                    v not in c_dict[u]
                    and u != v
                    and not would_cause_cycle(c_dict, u, v)
                ):
                    # FOR MMHC ALGORITHM -> Edge Restrictions
                    if restriction is None or (u, v) in restriction:
                        # SCORE FOR 'V' -> gaining a parent
                        old_cols = (v,) + tuple(p_dict[v])  # without 'u' as parent
                        mi_old = mutual_information(data[:, old_cols])
                        new_cols = old_cols + (u,)  # with'u' as parent
                        mi_new = mutual_information(data[:, new_cols])
                        delta_score = nrow * (mi_old - mi_new)

                        if delta_score > max_delta:
                            # if debug:
                            # 	print('Improved Arc Addition: ' , (u,v))
                            # 	print('Delta Score: ' , delta_score)
                            max_delta = delta_score
                            max_operation = "Addition"
                            max_arc = (u, v)

        ### TEST ARC DELETIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u]:
                    # SCORE FOR 'V' -> losing a parent
                    old_cols = (v,) + tuple(p_dict[v])  # with 'u' as parent
                    mi_old = mutual_information(data[:, old_cols])
                    new_cols = tuple(
                        [i for i in old_cols if i != u]
                    )  # without 'u' as parent
                    mi_new = mutual_information(data[:, new_cols])
                    delta_score = nrow * (mi_old - mi_new)

                    if delta_score > max_delta:
                        # if debug:
                        # 	print('Improved Arc Deletion: ' , (u,v))
                        # 	print('Delta Score: ' , delta_score)
                        max_delta = delta_score
                        max_operation = "Deletion"
                        max_arc = (u, v)

        ### TEST ARC REVERSALS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u] and not would_cause_cycle(c_dict, v, u, reverse=True):
                    # SCORE FOR 'U' -> gaining 'v' as parent
                    old_cols = (u,) + tuple(p_dict[v])  # without 'v' as parent
                    mi_old = mutual_information(data[:, old_cols])
                    new_cols = old_cols + (v,)  # with 'v' as parent
                    mi_new = mutual_information(data[:, new_cols])
                    delta1 = nrow * (mi_old - mi_new)
                    # SCORE FOR 'V' -> losing 'u' as parent
                    old_cols = (v,) + tuple(p_dict[v])  # with 'u' as parent
                    mi_old = mutual_information(data[:, old_cols])
                    new_cols = tuple(
                        [u for i in old_cols if i != u]
                    )  # without 'u' as parent
                    mi_new = mutual_information(data[:, new_cols])
                    delta2 = nrow * (mi_old - mi_new)
                    # COMBINED DELTA-SCORES
                    delta_score = delta1 + delta2

                    if delta_score > max_delta:
                        # if debug:
                        # 	print('Improved Arc Reversal: ' , (u,v))
                        # 	print('Delta Score: ' , delta_score)
                        max_delta = delta_score
                        max_operation = "Reversal"
                        max_arc = (u, v)

        ### DETERMINE IF/WHERE IMPROVEMENT WAS MADE ###
        if max_delta != 0:
            improvement = True
            u, v = max_arc
            if max_operation == "Addition":
                if debug:
                    print("ADDING: ", max_arc, "\n")
                c_dict[u].append(v)
                p_dict[v].append(u)
            elif max_operation == "Deletion":
                if debug:
                    print("DELETING: ", max_arc, "\n")
                c_dict[u].remove(v)
                p_dict[v].remove(u)
            elif max_operation == "Reversal":
                if debug:
                    print("REVERSING: ", max_arc, "\n")
                    c_dict[u].remove(v)
                    p_dict[v].remove(u)
                    c_dict[v].append(u)
                    p_dict[u].append(v)
        else:
            if debug:
                print("No Improvement on Iter: ", _iter)

        ### TEST FOR MAX ITERATION ###
        _iter += 1
        if _iter > max_iter:
            if debug:
                print("Max Iteration Reached")
            break

    # bn = BayesNet(c_dict)
    # print("bn is: " + str(bn.E))

    return c_dict


def optimal_network(Z, data):
    Z = sorted(Z)
    _, kVar = np.shape(data)
    DAG = np.zeros((kVar, kVar))
    data_array = np.array(data, dtype=np.int_)
    while kVar > 0:
        kVar -= 1
        if kVar not in Z:
            data_array = np.delete(data_array, kVar, axis=1)

    z_dict = hc(data_array, metric="BIC")
    c_dict = dict()
    for key, value in z_dict.items():
        if value == []:
            c_dict.setdefault(Z[key], [])
        else:
            c_list = []
            for i in value:
                c_list.append(Z[i])
                DAG[Z[key], Z[i]] = 1
            c_dict.setdefault(Z[key], c_list)
    return DAG


def S2TMB(data, target):
    # step 1:find the PC set
    _, kVar = np.shape(data)
    pc_t = []
    o_set = [i for i in range(kVar) if i != target]
    for x in o_set:
        Z = set([target, x]).union(pc_t)
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]

    # step2: remove false PC nodes and find spouses
    spouses_t = []
    varis_set = [i for i in range(kVar) if i != target and i not in pc_t]
    for x in varis_set:
        Z = set([target, x]).union(set(pc_t)).union(set(spouses_t))
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        spouses_t = [
            i
            for i in range(kVar)
            for j in range(kVar)
            if i != target and DAG[target, j] == 1 and DAG[i, j] == 1
        ]

    MB = list(set(pc_t).union(set(spouses_t)))
    return pc_t, MB


def s2tmb(data: np.ndarray, target: int, *wargs, **kwargs) -> set:
    """
    Find the Markov blanket of a target node in a graph using the S^2TMB algorithm [Gao and Ji, 2017].

    Args:
        data (np.ndarray): The data matrix.
        target (int): Target node for which to find the Markov blanket.
        *wargs: Additional positional arguments are ignored.
        **kwargs: Additional keyword arguments are ignored.
    Returns:
        set: The Markov blanket of the target node.
    """
    return set(S2TMB(data, target)[1])
