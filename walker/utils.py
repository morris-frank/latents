from itertools import tee

import numpy as np
import pandas as pd
from scipy import stats


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def linspace_gaussian(start, stop, num=50):
    return stats.norm.ppf(np.linspace(stats.norm.cdf(start), stats.norm.cdf(stop), num))


def load_script(name: str) -> pd.DataFrame:
    return pd.read_csv(
        f"./scripts/{name}.csv", header=None, names=["line", "cue"]
    ).sort_values("cue")
