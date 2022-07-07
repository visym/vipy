import vipy
import vipy.pyramid
import numpy as np
import itertools


def _show_foveation(im=None):
    assert im is None or isinstance(im, vipy.image.Image)    
    im = vipy.image.vehicles().centersquare().mindim(512).load() if im is None else im
    modes = ['log-circle', 'gaussian', 'linear-circle', 'linear-square']
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    return vipy.visualize.montage([vipy.pyramid.Foveation(im, mode=m, s=s)() for m in modes for s in scales], 256, 256, 4, 5).show().saveas('_test_foveation.png')


def _show_saccade(im=None):
    assert im is None or isinstance(im, vipy.image.Image)
    im = vipy.image.vehicles().centersquare().mindim(512).load() if im is None else im
    f = vipy.pyramid.Foveation(im, mode='log-circle')
    txy = list(np.arange(1,-1.5,-1/2))
    return vipy.visualize.montage([f(tx, ty) for (ty, tx) in itertools.product(txy, txy)], 256, 256, 5, 5).show().saveas('_test_saccade.png')

        
def test_laplacian():
    im = vipy.calibration.imcentersquare().rgb()
    assert np.allclose(vipy.pyramid.LaplacianPyramid(im).reconstruct().numpy(), im.numpy(), atol=20)
    assert np.allclose(vipy.pyramid.LaplacianPyramid(im, pad='reflect').reconstruct().numpy(), im.numpy(), atol=40)    
