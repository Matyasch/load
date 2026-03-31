import json
from pathlib import Path
import pickle

import numpy as np
import networkx as nx
from rpy2.robjects import r

r.source("generate_data.R")


def save_data(data: dict, seed: int, **kwargs):
    """
    Save experimental data to a file identified by the seed and the kwargs.

    Args:
        data (dict): The experimental data to save.
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.
    """
    file_path = Path("experiments/{}/{}.pkl".format(json.dumps(kwargs), seed))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file:
        pickle.dump(data, file)


def load_data(seed: int, **kwargs) -> dict:
    """
    Load experimental data from a file identified by the seed and the kwargs.

    Args:
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.

    Returns:
        dict: The loaded experimental data.
    """
    file_path = Path("experiments/{}/{}.pkl".format(json.dumps(kwargs), seed))
    with file_path.open("rb") as file:
        return pickle.load(file)


def generate_data(
    seed: int,
    nodes: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    connected: bool = True,
    expl_anc: bool = False,
    identifiable: bool = False,
    min_adj_size: int = 0,
    samples_num: int = 0,
    discrete: bool = False,
    max_classes: int = 2,
    file: str | None = None,
    save: bool = True,
) -> dict:
    """
    Generate experimental data given the parameters.

    Args:
        seed (int): The seed for the random number generator.
        file (str | None): The file to load data from.
        nodes (int): Number of nodes.
        exp_degree (float): Expected degree of the graph.
        max_degree (int): Maximum degree of the graph.
        targets (int): Number of target nodes.
        connected (bool): Whether the graph should be connected.
        expl_anc (bool): Whether the targets should be an explicit treatment-outcome pair.
        identifiable (bool): Whether the graph should be identifiable.
        min_adj_size (int): Minimum size of the adjacency set.
        samples_num (int): Number of samples.
        discrete (bool): Whether the data is discrete.
        max_classes (int): Maximum number of classes for discrete variables.

    Returns:
        dict: The generated experimental data.
    """
    if file is None:  # Generate causal model from scratch
        gen_func = r["generate_data"]
        kwargs = {
            "seed": seed,
            "nodes": nodes,
            "exp_degree": exp_degree,
            "max_degree": max_degree,
            "targets": targets,
            "connected": connected,
            "expl_anc": expl_anc,
            "identifiable": identifiable,
            "min_adj_size": min_adj_size,
            "samples_num": samples_num,
            "discrete": discrete,
            "max_classes": max_classes,
        }
    else:  # Generate data according to model from file
        gen_func = r["generate_data_from_file"]
        kwargs = {
            "file": file,
            "seed": seed,
            "targets": targets,
            "identifiable": identifiable,
            "min_adj_size": min_adj_size,
            "samples_num": samples_num,
        }
    try:  # Try to load previously generated experimental data
        return load_data(**kwargs)
    except FileNotFoundError:  # Generate new experimental data
        r_data = gen_func(**kwargs)
        true_dag = r_data.rx2("suffStat").rx2("g")
        true_dag = nx.from_numpy_array(
            np.array(r["as"](true_dag, "matrix")),
            create_using=nx.DiGraph,
        )
        dm = np.array(r_data.rx2("suffStat").rx2("dm"))
        data = {
            "id": r_data.rx2("id")[0],
            "data": dm,
            "targets": np.array(r_data.rx2("targets")).astype(np.int32) - 1,
            "true_dag": true_dag,
            "cpt": r_data.rx2("cpt"),
            "treatment": np.array(r_data.rx2("treatment")).astype(np.int32)[0] - 1,
            "outcome": np.array(r_data.rx2("outcome")).astype(np.int32)[0] - 1,
        }
        if save:
            save_data(data, **kwargs)
        return data
