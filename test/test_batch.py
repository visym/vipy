import os
import vipy.batch
import numpy as np
import vipy
from vipy.image import ImageDetection

rgbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_rgb.jpg')

def test_batch():
    vipy.globals.dask(num_processes=1)
    imb = vipy.batch.Batch([ImageDetection(filename=rgbfile, category='face', bbox=vipy.geometry.BoundingBox(0,0,100,100)) for k in range(0,100)])

    v = vipy.video.RandomScene()
    b = vipy.batch.Batch([v], n_processes=2)
    res = b.map(lambda v,k: v[k], args=[(k,) for k in range(0,len(v))]).result()
    assert isinstance(res[0], vipy.image.Scene)
    print('[test_image.batch]: batch  PASSED')

if __name__ == "__main__":
    test_batch()
    
