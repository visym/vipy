import numpy as np
from vipy.util import isnumber


def random_positive_semidefinite_matrix(N):
    """Return a randomly generated numpy float64 positive semidefinite matrix of size NxN"""
    assert isnumber(N), "Invalid input"""
    A = np.random.rand(N,N)
    return np.dot(A,A.transpose())


def column_stochastic(X, eps=1E-16):
    """Given a numpy array X of size MxN, return column stochastic matrix such that each of N columns sum to one.
     
    Args:
        X: [numpy] A 2D array
        eps: [float] a small floating point value to avoid divide by zero

    Returns:
        Matrix X such that columns sum to one.
    """
    x = np.sum(X, axis=0)
    return X / (eps + x.reshape((1, x.size)))


def row_stochastic(X, eps=1E-16):
    """Given a numpy array X of size MxN, return row stochastic matrix such that each of M rows sum to one.
     
    Args:
        X: [numpy] A 2D array
        eps: [float] a small floating point value to avoid divide by zero

    Returns:
        Matrix X such that rows sum to one.
    """
    x = np.sum(X, axis=1)
    return X / (eps + x.reshape((x.size, 1)))


def rowstochastic(X, eps=1E-16):
    """Alias for `vipy.linalg.row_stochastic`"""
    return row_stochastic(X, eps)


def bistochastic(X, numIterations=10, eps=1E-16):
    """Given a square numpy array X of size NxN, return bistochastic matrix such that each of N rows and N columns sum to one.

    Bistochastic matrix (doubly stochastic matrix) using Sinkhorn normalization.
     
    Args:
        X: [numpy] A square 2D array
        eps: [float] a small floating point value to avoid divide by zero
        numIterations: [int] The number of sinkhorn normalization iterations to apply

    Returns:
        Bistochastic matrix X
    """
    assert X.shape[0] == X.shape[1]  # square only
    for k in range(0, int(numIterations)):
        X = column_stochastic(row_stochastic(X, eps), eps)
    return(X)


def rectangular_bistochastic(X, numIterations=10):
    """Given a rectangular numpy array X of size MxN, return bistochastic matrix such that each of M rows sum to N/M and each if N columns sum to 1.

    Bistochastic matrix using Sinkhorn normalization on rectangular matrices
     
    Args:
        X: [numpy] A 2D array
        eps: [float] a small floating point value to avoid divide by zero
        numIterations: [int] The number of sinkhorn normalization iterations to apply

    Returns:
        Rectangular bistochastic matrix X
    """
    r = np.ones((X.shape[0],1))
    for k in range(0, int(numIterations)):
        c = 1.0 / (X.transpose().dot(r) + 1E-16)
        r = 1.0 / (X.dot(c) + 1E-16)
    return np.multiply(np.multiply(r, X), c.transpose())  # diag(r) * X * diag(c)


def row_normalized(X):
    """Given a rectangular numpy array X of size MxN, return a matrix such that each row has unit L2 norm.

    Args:
        X: [numpy] A 2D array

    Returns:
        Row normalized matrix X such that np.linalg.norm(X[i]) == 1, for all rows i
    """    
    for (k,x) in enumerate(X):
        X[k,:] = x / (np.linalg.norm(x.astype(np.float64)) + 1E-16)
    return(X)


def row_ssqrt(X):
    """Given a rectangular numpy array X of size MxN, return a matrix such that each element is the signed square root of the element in X.

    Args:
        X: [numpy] A rectangular 2D array 

    Returns:
        Matrix M such that elements M[i,j] preserve the sign of corresponding element in X, but the value is M[i,j] = sign(X[i,j]) * sqrt(abs(X[i,j]))
    """    
    for (k,x) in enumerate(X):
        x_L1 = x / (np.sum(np.abs(x.astype(np.float64))) + 1E-16)  # L1 normalized
        X[k,:] = np.multiply(np.sign(x_L1), np.sqrt(np.abs(x_L1)))  # signed square root
    return(X)


def normalize(x):
    """Given a numpy vector X of size N, return a vector with unit norm.

    Args:
        X: [numpy] A 1D array or a 2D array with one dim == 1

    Returns:
        Unit L2 norm of x, flattened to 1D
    """    

    x = x if x.ndim == 1 else (x.flatten() if x.ndim == 2 and min(x.shape) == 1 else None)
    assert x is not None, "Must be vector"                               
    return x / (np.linalg.norm(x.astype(np.float64)) + 1E-16)


def vectorize(X):
    """Convert a tuple X=([1], [2,3], [4,5,6]) to a numpy vector [1,2,3,4,5,6].
    
    Args:
        X: [list of lists, tuple of tuples] 

    Returns:
        1D numpy array with all elements in X stacked horizontally
    """
    return np.hstack(X).flatten()


def columnvector(x):
    """Convert a tuple with N elements to an Nx1 column vector"""
    z = vectorize(x)
    return z.reshape((z.size, 1))


def columnize(x):
    """Convert a numpy array into a flattened Nx1 column vector"""
    return x.flatten().reshape(x.size, 1)


def rowvector(x):
    """Convert a tuple with N elements to an 1xN row vector"""
    z = vectorize(x)
    return z.reshape((1, z.size))


def is_poweroftwo(x):
    """Is the number x a power of two?  
    
    >>> assert vipy.linalg.is_poweroftwo(4) == True
    >>> assert vipy.linalg.is_poweroftwo(3) == False
    """

    return x > 1 and ((x & (x - 1)) == 0)


def ndmax(A):
    """Return the (i,j,...)=(row, col,...) entry corresponding to the maximum element in the nd numpy matrix A
    
    >>> A = np.array([[1,2,3],[4,100,6]])
    >>> assert vipy.linalg.ndmax(A) == (1,2)
    
    """
    return np.unravel_index(A.argmax(), A.shape)


def ndmin(A):
    """Return the (i,j,...)=(row,col,...) entry corresponding to the minimum element in the nd numpy matrix A
    
    >>> A = np.array([[1,2,3],[4,100,6]])
    >>> assert vipy.linalg.ndmin(A) == (0,0)
    
    """
    return np.unravel_index(A.argmin(), A.shape)
