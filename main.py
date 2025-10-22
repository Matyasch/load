from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import pickle
import sys
from time import perf_counter

from causallearn.utils.cit import CIT
import numpy as np
from tqdm import tqdm

from algorithms import ALGORITHMS, snap
from generate_data import generate_data


class CountingTest:
    """
    Wrapper for CI tests that counts the number of tests done.
    """

    def __init__(
        self,
        data: np.ndarray,
        ci_test: str,
        **kwargs,
    ):
        self.cit = CIT(data, ci_test, **kwargs)
        if ci_test == "fisherz":
            self.cit.precision_matrix = np.linalg.inv(self.cit.correlation_matrix)

        self.method = self.cit.method
        self.tests_done = defaultdict(set)

    def __call__(
        self, X: int, Y: int, condition_set: list[int] | None = [], *args, **kwargs
    ) -> float:
        """
        Run the conditional independence test.

        Args:
            X (int): The first variable.
            Y (int): The second variable.
            condition_set (list[int] | None): The conditioning set.

        Returns:
            float: The p-value of the CI test.
        """
        if condition_set is None:
            condition_set = []
        self.tests_done[frozenset((X, Y))] |= {tuple(condition_set)}
        return self.cit(X, Y, condition_set)

    def get_tests_per_order(self) -> np.ndarray:
        """
        Get the number of tests done per order.

        Returns:
            np.ndarray: The number of tests done per order.
        """
        num_nodes = self.cit.data.shape[1]
        cond_sets = self.tests_done.values()
        if not cond_sets:
            return np.zeros(num_nodes, dtype=int)
        orders, test_num = np.unique(
            [len(cond) for conds in cond_sets for cond in conds],
            return_counts=True,
        )
        tests_per_order = np.zeros(num_nodes, dtype=int)
        tests_per_order[orders] = test_num
        return tests_per_order


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp", type=int, default=100, help="Number of experiments.")
    parser.add_argument("--file", type=str, help="Data file path.")
    parser.add_argument("--observed", type=int, help="Number of observed nodes.")
    parser.add_argument("--latent", type=int, default=0, help="Number of latent nodes.")
    parser.add_argument("--targets", type=int, default=4, help="Number of targets.")
    parser.add_argument(
        "--exp-degree", type=float, default=3.0, help="Expected degree."
    )
    parser.add_argument("--max-degree", type=int, default=10, help="Maximum degree.")
    parser.add_argument(
        "--connected", action="store_true", help="Generate connected graphs."
    )
    parser.add_argument(
        "--expl-anc", action="store_true", help="Provide treatment-outcome relation."
    )
    parser.add_argument(
        "--identifiable",
        action="store_true",
        help="Ensure that all targets are identifiable.",
    )
    parser.add_argument(
        "--min-adj-size",
        type=int,
        default=0,
        help="Minimum adjustment set size to identify (non-zero) causal effects.",
    )
    parser.add_argument(
        "--citest",
        type=str,
        help="Conditional independence test.",
    )
    parser.add_argument("--samples", type=int, default=0, help="Number of samples.")
    parser.add_argument(
        "--discrete", action="store_true", help="Generate discrete data."
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS.keys(),
        help="Causal discovery algorithm.",
    )
    parser.add_argument(
        "--mb-algorithm",
        type=str,
        help="The algorithm to use for finding the Markov blanket.",
        default="grow_shrink",
        choices=["grow_shrink", "total_conditioning"],
    )
    parser.add_argument(
        "--filter", action="store_true", help="Pre-filter with SNAP(0)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed.")
    parser.add_argument("--cont", type=str, help="Wandb ID of run to continue.")
    args = parser.parse_args(sys.argv[1:])
    return args


def run_algorithm(
    id: str,
    data: np.ndarray,
    targets: list[int],
    algorithm: str,
    ci_test: str,
    alpha: float,
    filter: bool = False,
    **kwargs,
):
    """
    Runs the specified algorithm on the provided experimental data.

    Args:
        id (str): The experiment ID.
        data (np.ndarray): The experimental data.
        targets (list[int]): The target variables.
        algorithm (str): The algorithm to run.
        ci_test (str): Name of the conditional independence test to use.
        alpha (float): The significance level for the CI tests.
        filter (bool): Whether to pre-filter with SNAP(0).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: The result of the algorithm and other statistics.
    """
    ci_test = CountingTest(data, ci_test, **kwargs)
    start = perf_counter()
    # Run algorithm
    try:
        if filter:
            res = snap(data, ci_test, 0.05, targets, 0)
            ignore = np.setdiff1d(np.arange(data.shape[1]), res["poss_anc"])
        else:
            ignore = []
        result = ALGORITHMS[algorithm](
            data=data,
            ci_test=ci_test,
            alpha=alpha,
            targets=targets,
            ignore=ignore,
            **kwargs,
        )
        result["failed"] = False
    except Exception:
        result = {"failed": True}
    # Log statistics
    result["time"] = perf_counter() - start
    result["id"] = id
    result["targets"] = targets
    result["tests"] = ci_test.get_tests_per_order().tolist()
    return result


def experiment(args, s: int) -> dict:
    """
    Run a single experiment with a specific seed.

    Args:
        args (Namespace): The command line arguments.
        done (dict): A dictionary of completed experiments.
        s (int): The seed for the experiment.

    Returns:
        dict: The result of the experiment.
    """
    data = generate_data(
        seed=args.seed + s,
        file=args.file,
        observed=args.observed,
        latent=args.latent,
        exp_degree=float(args.exp_degree),
        max_degree=args.max_degree,
        targets=args.targets,
        connected=args.connected,
        identifiable=args.identifiable,
        min_adj_size=args.min_adj_size,
        samples_num=args.samples,
        discrete=args.discrete,
        max_classes=2,
        expl_anc=args.expl_anc,
    )
    if args.algorithm is None:
        return {}
    return run_algorithm(
        algorithm=args.algorithm,
        ci_test=args.citest,
        mb_algorithm=args.mb_algorithm,
        alpha=args.alpha,
        observed=args.observed,  # latent models
        filter=args.filter,
        **data,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    algorithm = args.algorithm
    project = algorithm if args.file is None else args.file

    # Run experiments
    workers = 1 if args.citest in ["fisherz", "kci"] else None
    processes = []
    with ProcessPoolExecutor(workers) as exec:
        for s in range(args.exp):
            process = exec.submit(experiment, args, s)
            processes.append(process)

        if algorithm is None:  # Only generating data
            for p in tqdm(
                as_completed(processes), total=len(processes), desc="Generated data"
            ):
                pass
        else:  # Also running algorithms
            results = []
            for p in tqdm(
                as_completed(processes),
                total=len(processes),
                desc="Completed algorithm",
            ):
                results.append(p.result())
            with open(f"results_{project}.pkl", "wb") as f:
                pickle.dump(results, f)
