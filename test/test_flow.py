import vipy
from vipy.flow import Flow
import numpy as np

def test_flow():
    imfrom = vipy.image.RandomScene(num_objects=1)
    imto = imfrom.clone().zeropad(5, 10).cornercrop(imfrom.height(), imfrom.width())
    imf = Flow().imageflow(imfrom, imto)
    assert np.abs(np.median(imf.dx()) - 5) < 1 and np.abs(np.median(imf.dy()) - 10) < 1
    print('[test_flow.imageflow]: PASSED')

if __name__ == "__main__":
    test_flow()
    


