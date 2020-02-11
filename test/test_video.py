import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl
from vipy.geometry import BoundingBox
import pdb
from vipy.dataset.kinetics import Kinetics400, Kinetics600, Kinetics700
from vipy.dataset.activitynet import ActivityNet
from vipy.dataset.lfw import LFW


mp4file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video.mp4')
mp4url = 'https://www.youtube.com/watch?v=PYOSKYWg-5E'


def _test_dataset():
    d = Kinetics400('/tmp/kinetics').download().trainset()
    v = d[0].load()[0].resize(rows=256).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics400 PASSED')
    
    d = Kinetics600('/tmp/kinetics').download().trainset()
    v = d[1].load()[0].resize(rows=256).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics600 PASSED')
    
    d = Kinetics700('/tmp/kinetics').download().trainset()
    v = d[2].load()[0].rescale(0.5).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics700 PASSED')
    
    d = ActivityNet('/tmp/activitynet').download().dataset()
    v = d[0].load()
    print('[test_datasets]:  ActivityNet  PASSED')
    
    d = LFW('/tmp/lfw').download().dataset()
    d[0].saveas('lfw.jpg')
    print('Video.datasets: PASSED')


def test_video():
    # Common Parameters
    urls = vipy.videosearch.youtube('owl',1)
    assert isurl(urls[0])
    print('[test_video.video]: videosearch   PASSED')
    
    # Empty constructor
    try:
        v = vipy.video.Video()
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_video.video]: Empty constructor  PASSED')

    # Malformed URL should raise exception
    try:
        v = vipy.video.Video(url='myurl')
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_video.video]: Malformed URL constructor   PASSED')

    # Invalid constructors
    try:
        v = vipy.video.Video(array=np.random.rand(1,10,10,3))
        raise Failed()  # np.float64 not allowed
    except Failed:
        raise
    except:
        pass
    try:
        v = vipy.video.Video(array=np.random.rand(1,10,10,3).astype(np.float32), colorspace='rgb')
        raise Failed()  # rgb must be uint8
    except Failed:
        raise
    except:
        pass
    try:
        v = vipy.video.Video(array=np.random.rand(1,10,10,2).astype(np.uint8), colorspace='rgb')
        raise Failed()  # rgb must be uint8, three channel
    except Failed:
        raise
    except:
        pass
    try:
        v = vipy.video.Video(array=np.random.rand(1,10,10,3).astype(np.uint8), colorspace='lum')
        raise Failed()  # lum must be one channel
    except Failed:
        raise
    except:
        pass

    # Valid URL should not raise exception (even if it is not an image extension)
    im = vipy.video.Video(url='http://visym.com')
    print('[test_video.video]: Image URL constructor  PASSED')
        
    # Valid constructors
    v = vipy.video.Video(url=mp4url)
    v.__repr__()    
    v = vipy.video.Video(filename=mp4file)
    v.__repr__()
    v = vipy.video.Video(filename=mp4file, url=mp4url)
    v.__repr__()
    v = vipy.video.Video(array=np.random.rand(1,10,10,5).astype(np.float32), colorspace='float')
    v.__repr__()
    v = vipy.video.Video(array=np.random.rand(1,10,10,3).astype(np.uint8), colorspace='rgb')
    v.__repr__()
    v = vipy.video.Video(array=np.random.rand(5,10,10,3).astype(np.uint8), colorspace='bgr')
    v.__repr__()
    v = vipy.video.Video(array=np.random.rand(5,10,10,1).astype(np.uint8), colorspace='lum')
    v.__repr__()
    print('[test_video.video]: __repr__  PASSED')    
    
    
def _test_scene():

    # Downloader
    v = vipy.video.Video(url='http://visym.com/out.mp4').load(ignoreErrors=True)
    print('[test_video.video]: download ignoreErrors  PASSED')                
    v = vipy.video.Scene(url=mp4url).trim(0,100).load()
    print('[test_video.scene: download  PASSED')        
    for im in v:
        assert im.shape() == v.shape()
    print('[test_video.scene]: __iter__  PASSED')
    
    
    vid = vipy.video.Scene(filename=mp4file, tracks=[vipy.object.Track('person', frames=[0,200], boxes=[BoundingBox(xmin=0,ymin=0,width=200,height=400), BoundingBox(xmin=0,ymin=0,width=400,height=100)]),
                                                     vipy.object.Track('vehicle', frames=[0,200], boxes=[BoundingBox(xmin=100,ymin=200,width=300,height=400), BoundingBox(xmin=400,ymin=300,width=200,height=100)])])

    v = vid.clone().trim(0,10).load()
    assert len(v) == 10
    v.flush().trim(10,20).load()
    assert len(v) == 10
    print('[test_video.scene]:  trim: PASSED')

    (h,w) = v.shape()
    v = vid.clone().trim(0,10).rot90ccw().load()
    assert v.width() == h and v.height() == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 300]
    print('[test_video.scene]: rot90ccw  PASSED')

    v = vid.clone().trim(0,10).rot90cw().load()
    assert v.width() == h and v.height() == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 300]    
    print('[test_video.scene]: rot90cw  PASSED')

    v = vid.clone().trim(0,10).rescale(0.5).load(verbosity=0)
    assert v.height() * 2 == h and v.width() * 2 == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 200]
    assert [im.crop().width() for im in v[0]] == [100, 150]        
    print('[test_video.scene]: rescale  PASSED')

    (H,W) = vid.clone().trim(0,200).load(verbosity=0).shape()
    v = vid.clone().trim(0,200).resize(cols=100).load(verbosity=0)    
    assert v.width() == 100 and len(v) == 200
    assert np.allclose([im.bbox.height() for im in v[0]], [400*(100.0/W), 400*(100.0/W)])
    assert np.allclose([im.bbox.width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])
    print('[test_video.scene]: resize isotropic  PASSED')
    
    v = vid.clone().resize(cols=100, rows=100).trim(0,11).load(verbosity=0)    
    assert v.width() == 100 and v.height() == 100 and len(v) == 11
    assert np.allclose([im.bbox.height() for im in v[0]], [400*(100.0/H), 400*(100.0/H)])
    assert np.allclose([im.bbox.width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])    
    print('[test_video.scene]: resize anisotropic   PASSED')
    
    v = vid.clone().trim(0,200).rot90cw().resize(rows=200).load(verbosity=0)
    assert v.height() == 200 and len(v) == 200
    print('[test_video.scene]: rotate and resize  PASSED')
    
    v.annotate('vipy.mp4')
    print('[test_video.scene]: annotate  PASSED')

    v = vid.clone().trim(150,200).rot90cw().resize(rows=200).load(verbosity=0)
    assert v.height() == 200 and len(v) == 50
    print('[test_video.scene]: trim, rotate, resize  PASSED')

    v = vid.clone().crop(BoundingBox(xmin=1, ymin=2, width=10, height=20)).load(verbosity=0)
    assert v.height() == 20 
    assert v[0][0].bbox.xmin() == -1 and v[0][0].bbox.ymin() == -2
    assert v[0][1].bbox.xmin() == 99 and v[0][1].bbox.ymin() == 198
    print('[test_video.scene]: crop  PASSED')

    # Bounding box interpolation
    for im in v:
        assert im.shape() == v.shape()
    assert im[0].bbox.width() == 400 and im[1].bbox.width() == 200
    print('[test_video.scene]: frame interpolation  PASSED')

    # Thumbnail
    v = vid.clone().thumbnail()
    assert os.path.exists(v)
    print('[test_video.scene]: thumbnail ("%s")  PASSED' % v)
    
if __name__ == "__main__":
    test_video()
    _test_scene()
    _test_dataset()    
