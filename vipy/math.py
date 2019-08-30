import numpy as np
import scipy.ndimage

def runningmean(X, n):
    """Compute the running unweighted mean of X row-wise, with a history of n, with reflection along each column"""
    return scipy.ndimage.uniform_filter1d(X.astype(np.float32), axis=0, size=n, mode='reflect')





