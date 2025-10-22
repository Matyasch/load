from contextlib import chdir
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from causallearn.utils.cit import CIT

pytetrad_path = Path.cwd() / "py-tetrad/pytetrad"
if pytetrad_path.exists():
    sys.path.insert(1, str(pytetrad_path))
    with chdir("py-tetrad/pytetrad"):
        import tools.TetradSearch as ts
else:
    print("Warning: py-tetrad not found")


def fges(
    data: np.ndarray,
    ci_test: CIT,
    **kwargs,
) -> dict:
    """
    FGES algorithm.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (CIT): CI test to determine appropriate score function.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned CPDAG.
    """
    # Prepare data
    if ci_test.method == "chisq":
        data = data.astype(int)
    data = pd.DataFrame(data)
    # Run FGES
    search = ts.TetradSearch(data)
    search.set_verbose(False)
    if ci_test.method == "fisherz":
        search.use_sem_bic()
    elif ci_test.method == "chisq":
        search.use_bdeu()
    search.run_fges()
    # Convert result
    tetrad_amat = search.get_graph_to_matrix().to_numpy()
    tetrad_amat = tetrad_amat
    amat = np.zeros_like(tetrad_amat)
    amat[tetrad_amat == 2] = 1
    amat[np.logical_and(tetrad_amat == 3, tetrad_amat.T == 3)] = 1
    return {"amat": amat.tolist()}
