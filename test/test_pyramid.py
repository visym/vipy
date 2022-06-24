import vipy.pyramid
import numpy as np
import itertools


def _test_foveation():
    im = vipy.image.vehicles().mindim(512).load()
    f = vipy.pyramid.Foveation(im)

    for (ty, tx) in itertools.product(list(np.arange(-1,1,1/8)), list(np.arange(-1,1,1/8))):
        f(tx,ty).show()
        print(tx,ty)

def _test_laplacian():
    vipy.pyramid.LaplacianPyramid(vipy.image.owl().centersquare()).show()
