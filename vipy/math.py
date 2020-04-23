import numpy as np
import scipy.ndimage


def poweroftwo(x):
    """Return the closest power of two smaller than the value"""
    assert x>=2 
    return int(np.power(2, int(np.floor(np.log2(x)/np.log2(2)))))


def signsqrt(x):
    """Return the signed square root of elements in x"""
    return np.multiply(np.sign(x), np.sqrt(np.abs(x)))


def runningmean(X, n):
    """Compute the running unweighted mean of X row-wise, with a history of n, with reflection along each column"""
    return scipy.ndimage.uniform_filter1d(X.astype(np.float32), axis=0, size=n, mode='reflect')


def find_closest_positive_divisor(a, b):
    """Return non-trivial positive integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh == 0.  This uses exhaustive search, which is inefficient for large a."""
    assert a>0 and b>0
    if a<=b:
        return a
    for k in range(0, a-b+1):
        bh = b + k
        if bh>1 and a % bh == 0:
            return bh
        bh = b - k
        if bh>1 and a % bh == 0:
            return bh
    return a  # should never get here, since bh=a is always a solution
