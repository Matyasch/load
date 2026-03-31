from algorithms.ldecc import ldecc
from algorithms.ldecc_plus import ldecc_plus
from algorithms.ldp import ldp
from algorithms.ldp_plus import ldp_plus
from algorithms.load import load
from algorithms.load_oracle import load_oracle
from algorithms.marvel import marvel
from algorithms.mb_by_mb import mb_by_mb
from algorithms.mb_by_mb_plus import mb_by_mb_plus
from algorithms.pc import pc
from algorithms.snap import snap

ALGORITHMS = {
    "ldecc": ldecc,
    "ldecc_plus": ldecc_plus,
    "ldp": ldp,
    "ldp_plus": ldp_plus,
    "load": load,
    "load_oracle": load_oracle,
    "marvel": marvel,
    "mb_by_mb": mb_by_mb,
    "mb_by_mb_plus": mb_by_mb_plus,
    "pc": pc,
    "snap": snap,
}
