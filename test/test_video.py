import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir
from vipy.geometry import BoundingBox
import pdb
from vipy.dataset.kinetics import Kinetics400, Kinetics600, Kinetics700
from vipy.dataset.activitynet import ActivityNet
from vipy.dataset.lfw import LFW
from vipy.object import Detection, Track, Activity
import shutil


mp4file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video.mp4')
mp4url = 'https://www.youtube.com/watch?v=PYOSKYWg-5E'


def test_video():
    # Common Parameters
    urls = vipy.videosearch.youtube('owl',1)
    if len(urls) > 0:
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
    

    # Saveas
    v = vipy.video.Video(url=mp4url)
    v.save(ignoreErrors=True)
    print('[test_video.video]: saveas(ignoreErrors=True)  PASSED')        

    
    # Clone
    v = vipy.video.RandomVideo(64,64,64)
    vc = v.clone()
    assert np.allclose(vc._array, v._array)    
    vc._array = 0
    assert not np.allclose(vc._array, v._array)
    vc = v.clone(flushforward=True)
    assert vc._array is None and v._array is not None
    vc = v.clone(flushbackward=True)
    assert vc._array is not None and v._array is None
    vc = v.clone(flush=True)
    assert vc._array is None and v._array is None
    print('[test_video.scene]: clone  PASSED')        

    
def _test_scene():

    # Activityclip
    v = vipy.video.RandomSceneActivity(64,64,64)
    vc = v.clone(flushforward=True).filename('Video.mp4')
    assert vc._array is None and v._array is not None
    activitylength = [len(a) for a in vc.activities().values()]
    assert all([len(c.activities())==1 for c in vc.activityclip()])    
    assert all([len(a)==al for (c,al) in zip(vc.activityclip(padframes=0), activitylength) for a in c.activities().values()])
    try:
        vc.activityclip(padframes=2)  # will result in startframe < 0
        Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_video.scene]: activityclip  PASSED')    
    
    # Activitycrop
    v = vipy.video.RandomSceneActivity(64,64,64)
    vc = v.clone(flushforward=True).filename('Video.mp4')
    assert vc._array is None and v._array is not None
    a = vc.activitycrop()
    print('[test_video.scene]: activitycrop  PASSED (exercise only)')        
    
    # Downloader
    v = vipy.video.Video(url='http://visym.com/out.mp4').load(ignoreErrors=True)
    print('[test_video.video]: download ignoreErrors  PASSED')                
    v = vipy.video.Scene(url=mp4url).clip(0,100).load()
    print('[test_video.scene: download  PASSED')        
    for im in v:
        assert im.shape() == v.shape()
    print('[test_video.scene]: __iter__  PASSED')
    
    
    vid = vipy.video.Scene(filename=mp4file, tracks=[vipy.object.Track(category='person', keyframes=[0,200], boxes=[BoundingBox(xmin=0,ymin=0,width=200,height=400), BoundingBox(xmin=0,ymin=0,width=400,height=100)]),
                                                     vipy.object.Track(category='vehicle', keyframes=[0,200], boxes=[BoundingBox(xmin=100,ymin=200,width=300,height=400), BoundingBox(xmin=400,ymin=300,width=200,height=100)])])

    # Loader
    try:
        v = vid.clone().load(startframe=0)
        Failed()
    except Failed:
        raise
    except:
        pass
    try:
        v = vid.clone().load(endframe=0)
        Failed()
    except Failed:
        raise
    except:
        pass
    try:
        v = vid.clone().load(mindim=1, rescale=1)
        Failed()
    except Failed:
        raise
    except:
        pass
    try:
        v = vid.clone().load(rotation='an_invalid_string')
        Failed()
    except Failed:
        raise
    except:
        pass

    v = vid.clone().load(startframe=0, endframe=10)
    assert len(v) == 10
    assert np.allclose(vid.clone().load(startframe=0, endframe=10, rotation='rot90ccw').array(), vid.clone().clip(0,10).rot90ccw().load().array())
    assert np.allclose(vid.clone().load(startframe=1, endframe=11, rotation='rot90cw').array(), vid.clone().clip(1,11).rot90cw().load().array())
    assert np.allclose(vid.clone().load(startframe=2, endframe=12, mindim=10).array(), vid.clone().clip(2,12).mindim(10).load().array())
    assert np.allclose(vid.clone().load(startframe=3, endframe=12, rescale=0.2).array(), vid.clone().clip(3,12).rescale(0.2).load().array())
    print('[test_video.scene]:  load: PASSED')
    
    v = vid.clone().clip(0,10).load()
    assert len(v) == 10
    vc = v.clone(flushforward=True).clip(1,4).load()
    assert len(vc) == 3
    print('[test_video.scene]:  trim: PASSED')

    (h,w) = v.shape()
    v = vid.clone().clip(0,10).rot90ccw().load()
    assert v.width() == h and v.height() == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 300]
    print('[test_video.scene]: rot90ccw  PASSED')

    v = vid.clone().clip(0,10).rot90cw().load()
    assert v.width() == h and v.height() == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 300]    
    print('[test_video.scene]: rot90cw  PASSED')

    v = vid.clone().clip(0,10).rescale(0.5).load(verbose=False)
    assert v.height() * 2 == h and v.width() * 2 == w and len(v) == 10
    assert [im.crop().height() for im in v[0]] == [200, 200]
    assert [im.crop().width() for im in v[0]] == [100, 150]        
    print('[test_video.scene]: rescale  PASSED')

    (H,W) = vid.clone().clip(0,200).load(verbose=False).shape()
    v = vid.clone().clip(0,200).resize(cols=100).load(verbose=False)    
    assert v.width() == 100 and len(v) == 200
    assert np.allclose([im.bbox.height() for im in v[0]], [400*(100.0/W), 400*(100.0/W)])
    assert np.allclose([im.bbox.width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])
    print('[test_video.scene]: resize isotropic  PASSED')
    
    v = vid.clone().resize(cols=100, rows=100).clip(0,11).load(verbose=False)    
    assert v.width() == 100 and v.height() == 100 and len(v) == 11
    assert np.allclose([im.bbox.height() for im in v[0]], [400*(100.0/H), 400*(100.0/H)])
    assert np.allclose([im.bbox.width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])    
    print('[test_video.scene]: resize anisotropic   PASSED')
    
    v = vid.clone().clip(0,200).rot90cw().resize(rows=200).load(verbose=False)
    assert v.height() == 200 and len(v) == 200
    print('[test_video.scene]: rotate and resize  PASSED')
    
    v.annotate('vipy.mp4')
    print('[test_video.scene]: annotate  PASSED')

    v = vid.clone().clip(150,200).rot90cw().resize(rows=200).load(verbose=False)
    assert v.height() == 200 and len(v) == 50
    print('[test_video.scene]: trim, rotate, resize  PASSED')

    v = vid.clone().crop(BoundingBox(xmin=1, ymin=2, width=10, height=20)).load(verbose=False)
    assert v.height() == 20 
    assert v[0][0].bbox.xmin() == -1 and v[0][0].bbox.ymin() == -2
    assert v[0][1].bbox.xmin() == 99 and v[0][1].bbox.ymin() == 198
    print('[test_video.scene]: crop  PASSED')

    # Bounding box interpolation
    for im in v:
        assert im.shape() == v.shape()
    assert len(im) == 0  # strict boundary by default
    assert v[200][0].bbox.width() == 400 and v[200][1].bbox.width() == 200
    print('[test_video.scene]: frame interpolation  PASSED')

    # Thumbnail
    v = vid.clone().thumbnail()
    assert os.path.exists(v)
    print('[test_video.scene]: thumbnail ("%s")  PASSED' % v)

    # Map
    v = vid.clone().clip(0,10).map(lambda img: img*0)
    assert (np.sum(v.array().flatten()) == 0)
    print('[test_video.scene]: map ("%s")  PASSED' % v)
    
    # Shared array
    img = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Video(array=img)
    img[0,:,:,:] = 0
    assert np.sum(v.array()[0]) == 0
    img = np.random.rand(2,2,2,3).astype(np.float32)    
    v = v.array(img, copy=False)
    img[0,:,:,:] = 0    
    assert np.sum(v.array()[0]) == 0
    img = np.random.rand(2,2,2,3).astype(np.float32)        
    v = v.fromarray(img)
    img[0,:,:,:] = 0
    assert np.sum(v.array()[0]) != 0    
    print('[test_video.scene]: array by reference  PASSED')

    # Mutable iterator
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Video(array=frames)
    for im in v:
        im.numpy()[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Video(array=frames)
    for im in v.numpy():
        im[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    v = vipy.video.Video(mp4file).clip(0,10).load()
    for im in v.numpy():
        im[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    v = vipy.video.Video(mp4file).clip(0,10).load()
    for im in v:
        img = im.numpy()
        img[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    print('[test_video.scene]: mutable iterator  PASSED')    

    # Scene iterator
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Scene(array=frames)
    for im in v:
        v.add(Detection(0, 0, 0, 100, 100))
        v.add(Track(category=1, keyframes=[1], boxes=[BoundingBox(0,0,1,1)]))
        v.add([1,2,3,4], category='test')
    print('[test_video.scene]: scene iterator  PASSED')
    
    # Random scenes
    v = vipy.video.RandomVideo(64,64,64)
    v = vipy.video.RandomScene(64,64,64)
    vorig = vipy.video.RandomSceneActivity(64,64,64)
    print(vorig[0])
    print(vorig[0][0])
    print('[test_video.scene]: random scene  PASSED')
    
    # Video scenes
    v1 = vipy.video.Video('Video.mp4', startframe=50, endframe=100).load()
    v2 = vipy.video.Video('Video.mp4').clip(0,100).load()    
    assert v1[0] == v2[50]
    print('[test_video.scene]: video scenes  PASSED')    


    # Saveas 
    shutil.copy('Video.mp4', '/tmp/Video.mp4')
    v = vipy.video.Video(filename='/tmp/Video.mp4', startframe=0, endframe=10)
    v.save()
    assert v.hasfilename() and os.path.getsize('Video.mp4') != os.path.getsize(v.filename())
    print('[test_video.scene]: saveas()  PASSED')    


def test_track():
    t = Track(category=1, keyframes=[1,10], boxes=[BoundingBox(0,0,10,11), BoundingBox(1,1,11,12)])
    assert t.boundingbox().xywh() == (0,0,11,12)
    print('[test_video.track]: boundingbox()  PASSED')
    
    assert t.during(1) and t.during(10) and t.during(5)
    assert not t.during(11)
    print('[test_video.track]: during()  PASSED')    

    t.dict()
    print('[test_video.track]: dict()  PASSED')

    for d in t:
        assert d.width() == 10 and d.height() == 11
    assert d.xmin() == 1 and d.ymin() == 1
    print('[test_video.track]: interpolation  PASSED')
    
    
if __name__ == "__main__":
    test_video()
    test_track()
    _test_scene()
