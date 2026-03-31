from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import combinations, permutations

import networkx as nx
import numpy as np
from tqdm import tqdm

from rpy2.robjects import default_converter, globalenv, numpy2ri, r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging

from algorithms.mb_by_mb import get_locally_valid_parent_sets

rpy2_logger.setLevel(logging.ERROR)
r.source("evaluate.R")
r.source("generate_data.R")


# UTILS
def generate_samples_from_graph(
    exp: dict, samples_num: int = 1000, seed: int = 0
) -> np.ndarray:
    with (default_converter + numpy2ri.converter).context():
        suff_stat = r["generate_samples_from_graph"](
            amat=nx.to_numpy_array(exp["true_dag"]),
            targets=np.array(exp["targets"]) + 1,
            seed=seed,
            samples_num=samples_num,
            cpt=exp["cpt"],
        )
        return np.array(suff_stat[0])


def generate_samples_from_graphs(
    experiments: dict, samples_num: int = 1000, seed: int = 0
) -> dict:
    samples = {}

    proc = []
    with ProcessPoolExecutor() as e:
        for exp_id, exp in experiments.items():
            p = e.submit(generate_samples_from_graph, exp, samples_num, seed)
            p.exp_id = exp_id
            proc.append(p)
        for p in tqdm(as_completed(proc), desc="Generating samples", total=len(proc)):
            samples[p.exp_id] = p.result()
    return samples


def dag2cpdag(dag: np.ndarray) -> np.ndarray:
    with (default_converter + numpy2ri.converter).context():
        cpdag = r["dag2cpdag"](r["as"](dag != 0, "graphNEL"))
        cpdag = np.array(r["as"](cpdag, "matrix")).astype(np.int8)
    return cpdag


def get_true_cpdags(experiments):
    cpdags = {}
    proc = []
    with ProcessPoolExecutor() as e:
        for exp_id in experiments:
            dag = nx.to_numpy_array(experiments[exp_id]["true_dag"])
            p = e.submit(dag2cpdag, dag)
            p.exp_id = exp_id
            proc.append(p)
        for p in tqdm(as_completed(proc), desc="Generating CPDAGs", total=len(proc)):
            cpdags[p.exp_id] = p.result()
    return cpdags


# TESTS & TIME
def get_test_nums(results):
    test_sum = [np.sum(h["tests"]) for h in results]
    return np.array(test_sum)


def get_times(results):
    return np.array([h["time"] for h in results])


# ADJ-SETS
def get_optimal_adj_set(amat, treatment, outcome):
    if not is_ancestor(treatment, outcome, amat):
        return None
    try:
        with (default_converter + numpy2ri.converter).context():
            S = r["optAdjSet"](
                (r["as"](amat != 0, "graphNEL")), treatment + 1, outcome + 1
            )
        return set(np.array(S) - 1)
    except Exception:
        return None


def get_canonical_adj_set(amat: np.ndarray, treatment: int, outcome: int):
    if not is_ancestor(treatment, outcome, amat):
        return None
    try:
        with (default_converter + numpy2ri.converter).context():
            S = r["adjustment"](
                (amat.T != 0), "cpdag", treatment + 1, outcome + 1, "canonical"
            )
        return set(np.array(S) - 1)
    except Exception:
        return None


def get_true_osets(experiments):
    true_osets = {}

    cpdag_procs = []
    with ProcessPoolExecutor() as cdapg_e:
        for exp_id, exp in experiments.items():
            p = cdapg_e.submit(dag2cpdag, nx.to_numpy_array(exp["true_dag"]))
            p.exp_id = exp_id
            cpdag_procs.append(p)
        oset_procs = []
        with ProcessPoolExecutor() as oset_e:
            for cpdag_p in tqdm(
                as_completed(cpdag_procs), desc="Getting CPDAGs", total=len(cpdag_procs)
            ):
                cpdag = cpdag_p.result()
                exp_id = cpdag_p.exp_id
                exp = experiments[exp_id]
                targets = exp["targets"]
                if targets[0] in nx.ancestors(exp["true_dag"], targets[1]):
                    treatment, outcome = targets[0], targets[1]
                else:
                    treatment, outcome = targets[1], targets[0]
                p = oset_e.submit(get_optimal_adj_set, cpdag, treatment, outcome)
                p.exp_id, p.treatment, p.outcome = exp_id, treatment, outcome
                oset_procs.append(p)

            for p in tqdm(
                as_completed(oset_procs),
                desc=f"Getting true oset",
                total=len(oset_procs),
            ):
                true_osets[p.exp_id] = {
                    "treatment": p.treatment,
                    "outcome": p.outcome,
                    "oset": p.result(),
                }
    return true_osets


def get_locally_valid_sets(target: int, amat: np.ndarray) -> list:
    parents = np.logical_and(amat[:, target] != 0, amat[target, :] == 0)
    parents = np.where(parents)[0].tolist()
    unoriented = np.logical_and(amat[:, target] != 0, amat[target, :] != 0)
    unoriented = np.where(unoriented)[0]
    skeleton = (amat + amat.T) != 0
    np.fill_diagonal(skeleton, True)

    valid_sets = [parents]
    for l in range(1, len(unoriented) + 1):
        for comb in combinations(unoriented, l):
            candidate_set = list(comb) + parents
            if np.all(skeleton[comb, :][:, candidate_set]):
                valid_sets.append(candidate_set)
    return valid_sets


# EVALUATE OSET
def get_precision(true: set, pred: set):
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fp = len(pred - true)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def get_recall(true: set, pred: set):
    if true == pred:
        return 1.0
    tp = len(true.intersection(pred))
    fn = len(true - pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def get_f1(precision: float, recall: float):
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


def get_oset(project: str, h: dict, treatment: int, outcome: int):
    if project in ["pc", "fges", "marvel", "snap"]:
        est_oset = get_optimal_adj_set(np.array(h["amat"]), treatment, outcome)
        if est_oset is not None:
            est_osets = [est_oset]
        else:
            est_osets = None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets = eval(h["adj_sets"])
        if (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    elif project in ["ldp", "ldp_plus"]:
        results = eval(h["results"])
        if (treatment, outcome) in results and results[(treatment, outcome)][
            "vas_exists"
        ]:
            parts = results[(treatment, outcome)]
            est_osets = [
                set(parts["Z1"]),
                set(parts["Z1"] + parts["Z4"]),
                set(parts["Z1"] + parts["Z4"] + parts["Z5"]),
            ]
        else:
            est_osets = None
    elif project == "local_optimal" or project.startswith("load"):
        adj_sets = eval(h["adj_sets"])
        if h["identifiable"] and (treatment, outcome) in adj_sets:
            est_osets = adj_sets[(treatment, outcome)]
        else:
            est_osets = None
    else:
        raise ValueError("Invalid project")
    return est_osets


def evaluate_oset(algorithm, results, true_osets: dict):
    prec_scores = []
    rec_scores = []
    f1_scores = []

    proc = []
    with ProcessPoolExecutor() as e:
        for h in results:
            if "failed" in h and h["failed"]:
                prec_scores.append(0.0)
                rec_scores.append(0.0)
                f1_scores.append(0.0)
                continue
            exp_id = h["id"]
            treatment = true_osets[exp_id]["treatment"]
            outcome = true_osets[exp_id]["outcome"]
            p = e.submit(get_oset, algorithm, h, treatment, outcome)
            p.exp_id = exp_id
            proc.append(p)

        for p in tqdm(as_completed(proc), total=len(proc)):
            exp_id = p.exp_id
            est_osets = p.result()
            true_oset = true_osets[exp_id]["oset"]
            if true_oset is not None and est_osets is not None:
                precs = [get_precision(true_oset, s) for s in est_osets]
                recs = [get_recall(true_oset, s) for s in est_osets]
                f1s = [get_f1(p, r) for p, r in zip(precs, recs)]
                prec_scores.append(max(precs))
                rec_scores.append(max(recs))
                f1_scores.append(max(f1s))
            elif true_oset == est_osets:  # Both None
                prec_scores.append(1.0)
                rec_scores.append(1.0)
                f1_scores.append(1.0)
            else:
                prec_scores.append(0.0)
                rec_scores.append(0.0)
                f1_scores.append(0.0)

    return (
        np.sort(prec_scores)[5:95],
        np.sort(rec_scores)[5:95],
        np.sort(f1_scores)[5:95],
    )


# INTERVENTION DISTANCE
def is_ancestor(t1: int, t2: int, amat: np.ndarray) -> bool:
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


def true_linear_gaussian_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, **kwargs
) -> float:
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    amat = nx.to_numpy_array(true_dag)
    return sum(
        np.prod([amat[path[i], path[i + 1]] for i in range(len(path) - 1)])
        for path in nx.all_simple_paths(true_dag, treatment, outcome)
    )


def true_binary_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, cpt: object, **kwargs
) -> float:
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    with (default_converter + numpy2ri.converter).context():
        return globalenv["true_binary_effect"](
            int(treatment) + 1, int(outcome) + 1, nx.to_numpy_array(true_dag), cpt
        )[0]


def true_causal_effects(experiments: dict, family="gaussian") -> dict:
    if family == "gaussian":
        get_effect = true_linear_gaussian_effect
    elif family == "binary":
        get_effect = true_binary_effect
    else:
        raise ValueError("Invalid family")
    effects = {}
    for exp_id, exp in tqdm(experiments.items(), desc="Getting true effects"):
        effects[exp_id] = {}
        for t1, t2 in permutations(exp["targets"], 2):
            effects[exp_id][(t1, t2)] = get_effect(
                treatment=t1, outcome=t2, true_dag=exp["true_dag"], cpt=exp["cpt"]
            )
    return effects


def estimate_binary(samples, val, outcome, treatment, adj_set):
    # Filter rows based on the treatment value
    samples_val = samples[samples[:, treatment] == val]
    # Create the design matrix X
    X = np.column_stack(
        [np.ones(len(samples_val))] + [samples_val[:, idx] for idx in adj_set]
    )
    # Outcome variable (filtered by val)
    y = samples_val[:, outcome]
    # Solve the least squares problem X * beta = y
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Predict the outcome for all rows in the original samples using the estimated beta
    X_full = np.column_stack(
        [np.ones(len(samples))] + [samples[:, idx] for idx in adj_set]
    )
    y_pred = X_full @ beta
    return y_pred.mean()


def estimate_ate(
    treatment: int,
    outcome: int,
    adj_sets: list[list[int]],
    samples: np.ndarray,
    family: str = "gaussian",
) -> list[float]:
    effects = []
    for adj_set in adj_sets:
        adj_set = list(adj_set)
        if family == "gaussian":
            x = samples[:, [treatment] + adj_set]
            y = samples[:, outcome]
            A = np.vstack((x.T, np.ones(len(x))))
            effect = np.linalg.lstsq(A.T, y, rcond=None)[0][0]
            effects.append(effect)
        elif family == "binary":
            do_1 = estimate_binary(samples, 1, outcome, treatment, adj_set)
            do_0 = estimate_binary(samples, 0, outcome, treatment, adj_set)
            effects.append(do_1 - do_0)
    return effects


def get_adj_sets(project: str, h: dict, t1: int, t2: int):
    if "failed" in h and h["failed"]:
        return [[]]
    if project in ["pc", "fges", "marvel", "snap"]:
        amat = np.array(h["amat"], dtype=np.int8)
        if is_ancestor(t1, t2, amat):
            oset = get_optimal_adj_set(amat, t1, t2)
            if oset != None:  # identifiable
                return [oset]
            else:  # unidentifiable, fall back to local IDA
                amat = -amat.copy()
                amat[np.logical_and(amat == 0, amat.T == -1)] = 1
                return get_locally_valid_parent_sets(amat, t1, t2)
        else:
            return None
    elif project in ["mb_by_mb", "ldecc", "mb_by_mb_plus", "ldecc_plus"]:
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    elif project in ["ldp", "ldp_plus"]:
        results: dict = eval(h["results"])
        if (t1, t2) in results:
            parts = results[(t1, t2)]
            return [
                parts["Z1"],
                parts["Z1"] + parts["Z4"],
                parts["Z1"] + parts["Z4"] + parts["Z5"],
            ]
        else:
            return None
    elif project == "local_optimal" or project.startswith("load"):
        adj_sets: dict = eval(h["adj_sets"])
        return adj_sets.get((t1, t2), None)
    else:
        raise ValueError("Invalid project")


def estimate_ates(
    results,
    algorithm: str,
    samples: dict,
    family: str = "gaussian",
) -> np.ndarray:
    ates = {}
    adj_set_proc = []
    with ProcessPoolExecutor() as adj_set_e:
        for h in results:
            ates[h["id"]] = {}
            for t1, t2 in permutations(h["targets"], 2):
                p = adj_set_e.submit(get_adj_sets, algorithm, h, t1, t2)
                p.exp_id = h["id"]
                p.t1 = t1
                p.t2 = t2
                adj_set_proc.append(p)
        ate_procs = []
        with ProcessPoolExecutor() as ate_e:
            for adj_sets_p in tqdm(
                as_completed(adj_set_proc),
                total=len(adj_set_proc),
            ):
                adj_sets = adj_sets_p.result()
                if adj_sets == None:
                    ates[adj_sets_p.exp_id][(adj_sets_p.t1, adj_sets_p.t2)] = [0.0]
                else:
                    p = ate_e.submit(
                        estimate_ate,
                        treatment=adj_sets_p.t1,
                        outcome=adj_sets_p.t2,
                        adj_sets=adj_sets,
                        samples=samples[adj_sets_p.exp_id],
                        family=family,
                    )
                    p.exp_id = adj_sets_p.exp_id
                    p.t1 = adj_sets_p.t1
                    p.t2 = adj_sets_p.t2
                    ate_procs.append(p)
            for p in tqdm(
                as_completed(ate_procs),
                total=len(ate_procs),
            ):
                ates[p.exp_id][(p.t1, p.t2)] = p.result()
        return ates


def intervention_distance(
    est_ates: dict, true_ates: dict, aggr: str = "abs"
) -> np.ndarray:
    distances = []
    for exp in tqdm(est_ates, desc="Calculating intervention distances"):
        exp_dist = []
        for pair in est_ates[exp]:
            true_ate = true_ates[exp][pair]
            if aggr == "mse":
                dist = [(true_ate - est_ate) ** 2 for est_ate in est_ates[exp][pair]]
            elif aggr == "abs":
                dist = [np.abs(true_ate - est_ate) for est_ate in est_ates[exp][pair]]
            else:
                raise ValueError("Invalid aggregation")
            exp_dist.append(np.mean(dist))
        distances.append(np.mean(exp_dist))
    return np.array(distances)
