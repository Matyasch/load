# Local Optimal Adjustments Discovery (LOAD)

This is the official code repository for **[Local Causal Discovery for Statistically Efficient Causal Inference](https://arxiv.org/pdf/2510.14582)** (AISTATS 2026) by Mátyás Schubert, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains a minimal and portable implementation of LOAD. Check out the [aistats2026 branch](https://github.com/Matyasch/load/tree/aistats2026) to reproduce the results presented in the paper.

### Installation

```bash
pip install git+https://github.com/Matyasch/load.git
```

Or, to install from a local clone:

```bash
pip install -e .
```

The package depends only on `numpy`.

### Implementation

The LOAD algorithm is implemented in [`src/load/load_alg.py`](src/load/load_alg.py).

### Examples
The following minimal examples show how to run `load` and interpret its results. To run the examples below, also install `causal-learn` and `networkx`:

```bash
pip install causal-learn networkx
```

In the following example, LOAD identifies treatment `0` and outcome `2`, and that the causal effect is identifiable and can be estimated with optimal adjustment set `{3,4}`. LOAD outputs no adjustment sets for the causal effect of `2` on `0` as it is identified as zero.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Create graph
true_dag = nx.DiGraph([(0, 1), (1, 2), (4, 0), (4, 1), (3, 0), (3, 2)])
nx.draw_circular(true_dag, with_labels=True)
# Create CI test
dummy_data = np.zeros((0, 5))
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
# Run LOAD
result = load(set(range(5)), alpha=0.01, ci_test=ci_test, targets=[2, 0])
print(result)

>> {'adj_sets': {(0, 2): [{3, 4}]}, 'identifiable': True}

```
---
In the following example, LOAD identifies treatment `0` and outcome `2`, and that the causal effect is not identifiable. Thus, LOAD outputs the locally valid parent adjustment sets of `0`, which are `{}` and `{3}`. LOAD outputs no adjustment sets for the causal effect of `2` on `0` as it is identified as zero.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Create graph
true_dag = nx.DiGraph([(0, 1), (1, 2), (4, 1), (3, 0), (3, 2)])
nx.draw_circular(true_dag, with_labels=True)
# Create CI test
dummy_data = np.zeros((0, 5))
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
# Run LOAD
result = load(set(range(5)), alpha=0.01, ci_test=ci_test, targets=[2, 0])
print(result)

>> {'adj_sets': {(0, 2): [set(), {3}]}, 'identifiable': False}

```
---
In the following example, LOAD finds that the causal relation between the targets cannot be determined and hence the causal effect is not identifiable. Thus, LOAD outputs the locally valid parent adjustment sets for both treatment `2` with outcome `0`, and treatment `0` with outcome `2`.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Create graph
true_dag = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 1), (3, 2)])
nx.draw_circular(true_dag, with_labels=True)
# Create CI test
dummy_data = np.zeros((0, 4))
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
# Run LOAD
result = load(set(range(4)), alpha=0.01, ci_test=ci_test, targets=[2, 0])
print(result)

>> {'adj_sets': {(2, 0): [set(), {1}, {3}, {1, 3}], (0, 2): [set(), {1}, {3}, {1, 3}]}, 'identifiable': False}
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
