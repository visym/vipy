import vipy.pyramid
import numpy as np
import itertools


def _test_foveation():
    #im = vipy.image.vehicles().mindim(512).load()
    im = vipy.image.people().mindim(512).load()    
    f = vipy.pyramid.Foveation(im, s=2, mode='linear-circle')
    #f = vipy.pyramid.Foveation(im, s=0.15, mode='log-circle')    

    for (ty, tx) in itertools.product(list(np.arange(-1,1,1/8)), list(np.arange(-1,1,1/8))):
        f(tx,ty).show()
        print(tx,ty)

        
def test_laplacian():
    im = vipy.calibration.imcentersquare().rgb()
    assert np.allclose(vipy.pyramid.LaplacianPyramid(im).reconstruct().numpy(), im.numpy(), atol=20)
    assert np.allclose(vipy.pyramid.LaplacianPyramid(im, pad='reflect').reconstruct().numpy(), im.numpy(), atol=40)    
