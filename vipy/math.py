import numpy as np
import scipy.ndimage


def signsqrt(x):
    """Return the signed square root of elements in x"""
    return np.multiply(np.sign(x), np.sqrt(np.abs(x)))


def runningmean(X, n):
    """Compute the running unweighted mean of X row-wise, with a history of n, with reflection along each column"""
    return scipy.ndimage.uniform_filter1d(X.astype(np.float32), axis=0, size=n, mode='reflect')


def find_closest_divisor(a, b):
    """Return integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh == 0.  This uses exhaustive search, which is inefficient for large a."""
    assert a>=b
    for k in range(0, a-b+1):
        bh = b + k
        if a % bh == 0:
            return bh
        bh = b - k
        if bh>0 and a % bh == 0:
            return bh
    return a  # should never get here, since bh=a is always a solution
