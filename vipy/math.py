import numpy as np
from vipy.util import isnumpy, chunklistWithOverlap


try:
    from numba import njit, config  
    config.THREADING_LAYER = 'workqueue'    
    @njit(parallel=True, cache=True, nogil=True, fastmath=True)
    def normalize(arr, mean, std, scale):
        """Whiten the numpy array arr using the provided mean and standard deviation.

        - Uses numba acceleration since this is a common operation for preparing tensors.
        - Computes:  ((scale*arr) - mean)) / std
        
        Args:
            arr: [numpy] A numpy array
            mean: [numpy] A broadcastable mean vector
            std: [numpy]  A broadcastable std vector
            scale: [float] A scale factor to apply to arr before whitening (e.g. to scale from [0,255] to [0,1])

        Returns
            ((scale*arr) - mean)) / std

        .. notes:: Does not check that std > 0
        """
        return ((np.float32(scale)*arr.astype(np.float32)) - mean.flatten()) / std.flatten() 
except:
    def normalize(arr, mean, std, scale):
        """Whiten the numpy array arr using the provided mean and standard deviation.

        Computes:  (scale*(arr - mean)) / std

        Args:
            arr: [numpy] A numpy array
            mean: [numpy] A broadcastable mean vector
            std: [numpy]  A broadcastable std vector
            scale: [float] A scale factor to apply to arr before whitening (e.g. to scale to [0,1])

        Returns
            ((scale*arr) - mean)) / std

        .. notes:: Does not check that std > 0
        """
        return ((np.float32(scale)*arr.astype(np.float32)) - mean.flatten()) / std.flatten() 


try:
    from numba import njit, config  
    config.THREADING_LAYER = 'workqueue'    
    @njit(parallel=True, cache=True, nogil=True, fastmath=True)
    def gain(arr, scale):
        """Rescale the numpy array arr using the provided gain

        - Uses numba acceleration since this is a common operation for preparing tensors.
        - Computes:  scale*arr
        
        Args:
            arr: [numpy] A numpy array
            scale: [float] A scale factor to apply to arr before whitening (e.g. to scale from [0,255] to [0,1])

        Returns
            (scale*arr)

        """
        return np.float32(scale)*arr.astype(np.float32)
except:
    def gain(arr, scale):
        return (np.float32(scale)*arr.astype(np.float32))
    
    
def _normalize(arr, mean, std, scale):
    """Whiten the numpy array arr using mean and standard deviation (no parallelization)"""
    return ((np.float32(scale)*arr.astype(np.float32)) - mean.flatten()) / std.flatten() 
    

def significant_digits(f, n):
    """Return the float f rounded to n significant digits"""
    assert (isinstance(f, float) or isinstance(f, int)) and isinstance(n, int)
    return np.round(f*np.power(10,n))/np.power(10, n)

def iseven(x):
    """is the number x an even number?"""
    return x%2 == 0


def isodd(x):
    """is the number x an odd number?"""
    return x%2 != 0


def even(x, greaterthan=False):
    """Return the largest even integer less than or equal (or greater than if greaterthan=True) to the value"""
    x = int(np.round(x))
    return x if x%2 == 0 else (x+1 if greaterthan else x-1)


def poweroftwo(x):
    """Return the closest power of two smaller than the scalar value. x=511 -> 256, x=512 -> 512"""
    assert x>=2 
    return int(np.power(2, int(np.floor(np.log2(x)/np.log2(2)))))


def signsqrt(x):
    """Return the signed square root of elements in numpy array x"""
    return np.multiply(np.sign(x), np.sqrt(np.abs(x)))


def runningmean(X, n):
    """Compute the running unweighted mean of X row-wise, with a history of n, reducing the history at the start for column indexes < n"""
    assert isnumpy(X), "Input must be a np.array()"    
    return np.array([[np.mean(c) for c in chunklistWithOverlap(x, n, n-1)] for x in X])


def gaussian(M, std=1, sym=True):
    """1D gaussian window with M points.

    Replication of scipy.signal.gaussian
    """

    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def gaussian2d(mu, std, H, W):
    """2D float32 gaussian image of size (rows=H, cols=W) with mu=[x, y] and std=[stdx, stdy]"""
    img = np.zeros( (H,W), dtype=np.float32)
    (X,Y) = np.meshgrid(W,H)
    gx = ((1.0/np.sqrt(2*np.pi))*np.exp(-0.5*((np.arange(W)-mu[0])**2 / (std[0]**2)))).astype(np.float32)
    gy = ((1.0/np.sqrt(2*np.pi))*np.exp(-0.5*((np.arange(H)-mu[1])**2) / (std[1]**2))).astype(np.float32)
    return np.outer(gy,gx)


def interp1d(x, y):
    """Replication of scipy.interpolate.interp1d with assume_sorted=True, and constant replication of boundary handling.  Input must be sorted.

    Input: 1D numpy arrays y=f(x)
    Output: Lambda function f that will interpolate f(xi) -> yi
    
    """
    def ceil(x, at):
        k = np.argwhere(np.array(x)-at > 0)
        return len(x)-1 if len(k) == 0 else int(k.flatten()[0])
    
    #assert all(sorted(x) == x) and all(sorted(y) == y), "Input must be sorted"
    return lambda at: y[max(0, ceil(x,at)-1)] + float(y[ceil(x,at)] - y[max(0, ceil(x,at)-1)])*((at - x[max(0, ceil(x,at)-1)])/(1E-16+(x[ceil(x,at)] - x[max(0, ceil(x,at)-1)])))


def find_closest_positive_divisor(a, b):
    """Return non-trivial positive integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh == 0.  

    .. notes:: This uses exhaustive search, which is inefficient for large a.
    """
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
    raise  # should never get here, since bh=a is always a solution

def cartesian_to_polar(x, y):
    """Cartesian (x,y) coordinates to polar (radius, theta_radians) coordinates, theta in radians in [-pi,pi]"""
    return (np.sqrt(np.array(x)**2 + np.array(y)**2), np.arctan2(y, x))

def polar_to_cartesian(r, t):
    """Polar (r=radius, t=theta_radians) coordinates to cartesian (x=right,y=down) coordinates.  (0,0) is upper left of image"""
    return (np.multiply(r, np.cos(t)), np.multiply(r, np.sin(t)))
            
def rad2deg(r):
    """Radians to degrees"""
    return r*(180.0 / np.pi)

