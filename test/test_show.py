import vipy
from vipy.show import colorlist
from vipy.image import Scene
import numpy as np


def _test_colors():
    """Show boxes for each color in colorlist() to make sure they are visible"""
    c = colorlist()

    ims = Scene(array=np.uint8(255*np.random.rand(1024,1024,3)),
                colorspace='rgb',
                category='scene',
                objects=[vipy.object.Detection(c, xmin=np.random.randint(0, 1024 - 16), ymin=np.random.randint(0, 1024 - 16), width=64, height=64)
                         for c in colorlist()])
    
    d_category2color = {c:c for c in colorlist()}
    ims.show(d_category2color=d_category2color)

