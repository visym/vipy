import os
import vipy.batch
import numpy as np
import vipy
from vipy.image import ImageDetection
import vipy.globals

rgbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_rgb.jpg')

def test_batch():
    vipy.globals.dask(num_processes=1)
    imb = vipy.batch.Batch([ImageDetection(filename=rgbfile, category='face', xywh=vipy.geometry.BoundingBox(0,0,100,100).xywh()) for k in range(0,100)])

    vipy.globals.dask(num_processes=2)
    v = vipy.video.RandomScene()
    b = vipy.batch.Batch([v])

    res = b.map(lambda v: v).result()
    print(res)
    assert isinstance(res[0], vipy.video.Scene)

    vipy.globals.noparallel()
    
    print('[test_image.batch]: batch  PASSED')

if __name__ == "__main__":
    test_batch()
    
