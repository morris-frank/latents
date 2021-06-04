from itertools import tee
import numpy as np
from scipy import stats


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def linspace_gaussian(start, stop, num=50):
    return stats.norm.ppf(np.linspace(stats.norm.cdf(start), stats.norm.cdf(stop), num))
