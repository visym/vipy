import numpy as np
import scipy.ndimage
from vipy.util import isstring


def signsqrt(x):
    """Return the signed square root of elements in x"""
    return np.multiply(np.sign(x), np.sqrt(np.abs(x)))

def runningmean(X, n):
    """Compute the running unweighted mean of X row-wise, with a history of n, with reflection along each column"""
    return scipy.ndimage.uniform_filter1d(X.astype(np.float32), axis=0, size=n, mode='reflect')

def isnumber(x):
    """Is the input a python type of a number or a string containing a number?"""
    return isinstance(x, (int, float)) or (isstring(x) and isfloat(x))

def isfloat(x):
    """Is the input a float or a string that can be converted to float?"""
    try:
        float(x)
        return True
    except ValueError:
        return False

