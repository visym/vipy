import numpy as np
from vipy.util import isnumber


def random_positive_semidefinite_matrix(N):
    """Return a randomly generated float64 positive semidefinite matrix of size NxN"""
    assert isnumber(N), "Invalid input"""
    A = np.random.rand(N,N)
    return np.dot(A,A.transpose())


def column_stochastic(X, eps=1E-16):
    x = np.sum(X, axis=0)
    return X / (eps + x.reshape((1, x.size)))


def row_stochastic(X, eps=1E-16):
    x = np.sum(X, axis=1)
    return X / (eps + x.reshape((x.size, 1)))


def rowstochastic(X, eps=1E-16):
    x = X.sum(axis=1)
    return X / (eps + x.reshape((x.size, 1)))


def bistochastic(X, numIterations=10):
    """Sinkhorn normalization"""
    assert X.shape[0] == X.shape[1]  # square only
    for k in range(0, int(numIterations)):
        X = column_stochastic(row_stochastic(X))
    return(X)


def rectangular_bistochastic(X, numIterations=10):
    """Sinkhorn normalization for rectangular matrices"""
    r = np.ones((X.shape[0],1))
    for k in range(0, int(numIterations)):
        c = 1.0 / (X.transpose().dot(r) + 1E-16)
        r = 1.0 / (X.dot(c) + 1E-16)
    return np.multiply(np.multiply(r, X), c.transpose())  # diag(r) * X * diag(c)


def row_normalized(X):
    for (k,x) in enumerate(X):
        X[k,:] = x / (np.linalg.norm(x.astype(np.float64)) + 1E-16)
    return(X)


def row_ssqrt(X):
    for (k,x) in enumerate(X):
        x_L1 = x / (np.sum(np.abs(x.astype(np.float64))) + 1E-16)  # L1 normalized
        X[k,:] = np.multiply(np.sign(x_L1), np.sqrt(np.abs(x_L1)))  # signed square root
    return(X)


def normalize(x):
    return x / (np.linalg.norm(x.astype(np.float64)) + 1E-16)


def vectorize(X):
    """Convert a tuple X=([1], [2,3], [4,5,6]) to a vector [1,2,3,4,5,6]"""
    return np.hstack(X).flatten()


def columnvector(x):
    """Convert a tuple with N elements to an Nx1 column vector"""
    z = vectorize(x)
    return z.reshape((z.size, 1))


def columnize(x):
    return x.flatten().reshape(x.size, 1)


def rowvector(x):
    """Convert a tuple with N elements to an 1xN row vector"""
    z = vectorize(x)
    return z.reshape((1, z.size))


def poweroftwo(x):
    return x > 1 and ((x & (x - 1)) == 0)


def ndmax(A):
    return np.unravel_index(A.argmax(), A.shape)


def ndmin(A):
    return np.unravel_index(A.argmin(), A.shape)
