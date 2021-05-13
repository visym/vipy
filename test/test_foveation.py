import vipy.torch
import numpy as np
import itertools

def _test_foveation():
    im = vipy.image.vehicles().mindim(512).load()
    f = vipy.torch.Foveation(im)

    for (ty, tx) in itertools.product(list(np.arange(-1,1,1/8)), list(np.arange(-1,1,1/8))):
        f(tx,ty).show()
        print(tx,ty)

