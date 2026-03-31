# Local Optimal Adjustments Discovery (LOAD)

This is the official code repository for **[Local Causal Discovery for Statistically Efficient Causal Inference](https://arxiv.org/pdf/2510.14582)** (AISTATS 2026) by Mátyás Schubert, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains all code to reproduce the results presented in the paper. Check out the [main branch](https://github.com/Matyasch/load/tree/main) for a minimal and portable implementation of LOAD.

Experiments can be run using `main.py` with the appropriate parameters. Evaluation functions are implemented in `evaluate.py`.

### Dependencies
Python dependencies can be installed with `pip3 install -r requirements.txt`

R dependencies can be installed as follows
```R
install.packages("BiocManager")
BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
install.packages(c("pcalg", "igraph", "expm", "bnlearn", "dagitty"))
```

### Example
Run an experiment from the terminal:
```sh
python3 main.py --connected --nodes 100 --exp-degree 2. --citest d_separation --alpha 0.01 --expl-anc --targets 2 --algorithm load
```

Evaluate the experiment in python:
```py
import pickle
import numpy as np
from evaluate import *

if __name__ == "__main__":
    with open("results_load.pkl", "rb") as f:
        results = pickle.load(f)

    experiments = {}
    for s in range(100):
        path = f'experiments/{{"nodes": 100, "exp_degree": 2.0, "max_degree": 10, "targets": 2, "connected": true, "expl_anc": true, "identifiable": false, "min_adj_size": 0, "samples_num": 0, "discrete": false, "max_classes": 2}}/{s}.pkl'
        with open(path, "rb") as f:
            exp = pickle.load(f)
            experiments[exp["id"]] = exp

    tests = np.sort(get_test_nums(results))[5:95]
    print(f"Tests: {tests.mean()} ± {tests.std()}")
    times = np.sort(get_times(results))[5:95]
    print(f"Times: {times.mean()} ± {times.std()}")

    true_osets = get_true_osets(experiments)
    f1 = np.sort(evaluate_oset("load", results, true_osets)[2])[5:95]
    print(f"F1: {f1.mean()} ± {f1.std()}")

    true_effects = true_causal_effects(experiments, "gaussian")
    samples = generate_samples_from_graphs(experiments, 10000)
    effects = estimate_ates(results, "load", samples, "gaussian")
    distance = np.sort(intervention_distance(effects, true_effects))[5:95]
    print(f"Intervention distance: {distance.mean()} ± {distance.std()}")
```

### Citation
```bibtex
@inproceedings{schubert2026local,
  title     = {Local Causal Discovery for Statistically Efficient Causal Inference},
  author    = {M{\'a}ty{\'a}s Schubert and Tom Claassen and Sara Magliacane},
  booktitle = {The 29th International Conference on Artificial Intelligence and Statistics},
  year      = {2026},
  url       = {https://openreview.net/forum?id=FlWl20PFd7}
}
```
