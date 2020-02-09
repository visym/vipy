import numpy as np


def checkerboard(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows)"""
    img = None
    for i in range(0,nrows):
        row = np.hstack([float((j + i) % 2) * np.ones((dx,dy)) for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def tile(T, nrows=16, ncols=16):
    """Create a 2D tile pattern with texture T repeated (M,N) times"""
    img = None
    for i in range(0,nrows):
        row = np.hstack([T for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def greenblock(dx, dy):
    """Create a green color numpy image of size (dx,dx)"""
    img = np.zeros((dx,dy,3))
    img[:,:,1] = 1.0
    return img


def redblock(dx, dy):
    """Create a red color numpy image of size (dx,dx)"""
    img = np.zeros((dx,dy,3))
    img[:,:,0] = 1.0
    return img


def blueblock(dx, dy):
    img = np.zeros((dx,dy,3))
    img[:,:,2] = 1.0
    return img


def bayer(dx, dy, M=16, N=16):
    """Return an (M,N) tiled texture pattern of [blue, green, blue, green; green red green red; blue green blue green, green red green red] such that each subblock is (dx,dy)"""
    T = np.vstack([np.hstack([blueblock(dx,dy), greenblock(dx,dy), blueblock(dx,dy), greenblock(dx,dy)]),
                   np.hstack([greenblock(dx,dy), redblock(dx,dy), greenblock(dx,dy), redblock(dx,dy)]),
                   np.hstack([blueblock(dx,dy), greenblock(dx,dy), blueblock(dx,dy), greenblock(dx,dy)]),
                   np.hstack([greenblock(dx,dy), redblock(dx,dy), greenblock(dx,dy), redblock(dx,dy)])])
    return tile(T, M, N)


def dots(dx=16, dy=16, nrows=8, ncols=8):
    """Create a sequence of dots with strides (dx, dy) and image of size (dx*ncols,dy*nrows)"""

    imdot = np.zeros((dx,dy))
    imdot[int(np.floor(dx / 2.0)), int(np.floor(dy / 2.0))] = 1.0
    img = None
    for i in range(0,nrows):
        row = np.hstack([imdot for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def vertical_gradient(nrows, ncols):
    """Create a linear ramp image """
    return np.outer([(255.0 * (x / float(nrows))) for x in range(0,nrows)], np.ones((1,ncols))).astype(np.uint8)
