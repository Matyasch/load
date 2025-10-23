# Local Optimal Adjustments Discovery (LOAD)

<div align="center">

[![paper](https://img.shields.io/badge/paper-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/pdf/2510.14582)

</div>

This is the official code repository for **Local Causal Discovery for Statistically Efficient Causal Inference** by Mátyás Schubert, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains a minimal and portable implementation of LOAD. Check out the [aistats2026 branch](https://github.com/Matyasch/load/tree/aistats2026) to reproduce the results presented in the paper.

The LOAD algorithm is implemented in [`load.py`](load.py) and depends only on `numpy`, as listed in [`requirements.txt`](requirements.txt).

### Examples
The following minimal examples show how to run `load` and interpret its results. Please also install `causal-learn` to run them.

In the following example, LOAD outputs that the causal effect is identifiable for treatment `0` and outcome `1` and can be estimated with optimal adjustment set `{5,6}`. LOAD returns no adjustment sets for the causal effect of `1` on `0` as it is identified as zero.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Generate data
true_dag = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
)
dummy_data = np.zeros_like(true_dag)
true_dag = nx.DiGraph(true_dag)
nx.draw_circular(true_dag, with_labels=True)

# Run LOAD
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
result = load(data=dummy_data, alpha=0.01, ci_test=ci_test, targets=[1, 0])
print(result)

>> {'adj_sets': {(0, 1): [{5, 6}]}, 'identifiable': True}
```
---
In the following example, LOAD outputs that the causal effect is not identifiable. For treatment `4` and outcome `1`, the locally valid parent adjustment sets are `{2, 3}` and `{0, 2, 3}`. LOAD returns no adjustment sets for the causal effect of `1` on `4`, which means that it is identified as zero.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Generate data
true_dag = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
)
dummy_data = np.zeros_like(true_dag)
true_dag = nx.DiGraph(true_dag)
nx.draw_circular(true_dag, with_labels=True)

# Run LOAD
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
result = load(data=dummy_data, alpha=0.01, ci_test=ci_test, targets=[1, 4])
print(result)

>> {'adj_sets': {(4, 1): [{2, 3}, {0, 2, 3}]}, 'identifiable': False}
```
---
In the following example, LOAD outputs that the causal effect is not identifiable, and both targets might be treatments. Thus, LOAD outputs the locally valid parent adjustment sets for both treatment `4` and outcome `0`, and treatment `0` and outcome `4`.
```py
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx

from load import load

# Generate data
true_dag = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
)
dummy_data = np.zeros_like(true_dag)
true_dag = nx.DiGraph(true_dag)
nx.draw_circular(true_dag, with_labels=True)

# Run LOAD
ci_test = CIT(data=dummy_data, method="d_separation", true_dag=true_dag)
result = load(data=dummy_data, alpha=0.01, ci_test=ci_test, targets=[0, 4])
print(result)

>> {'adj_sets': {(0, 4): [{2, 3}], (4, 0): [{2, 3}]}, 'identifiable': False}
```

### Citation
```bibtex
@article{schubert2025local,
  title={Local Causal Discovery for Statistically Efficient Causal Inference},
  author={Schubert, M{\'a}ty{\'a}s and Claassen, Tom and Magliacane, Sara},
  journal={arXiv preprint arXiv:2510.14582},
  year={2025}
}
```
