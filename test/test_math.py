import scipy.interpolate
from vipy.math import interp1d
import numpy as np


def test_interp1d():
    x = sorted(range(0,100))
    y = sorted(np.random.rand(100))

    f1 = scipy.interpolate.interp1d(x,y)
    f2 = interp1d(x,y)

    assert np.allclose(f1(0), f2(0))
    assert np.allclose(f1(1.25), f2(1.25))
    
    print('[test_math]: interp1d passed')

    
if __name__ == '__main__':
    test_interp1d()
    
