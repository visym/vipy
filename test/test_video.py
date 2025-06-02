import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir, totempdir, tempMP4
from vipy.geometry import BoundingBox
from vipy.data.kinetics import Kinetics400, Kinetics600, Kinetics700
from vipy.data.activitynet import ActivityNet
from vipy.data.lfw import LFW
from vipy.object import Detection, Track
from vipy.activity import Activity
import shutil
from collections import Counter  

mp4file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video.mp4')
mp4url = 'https://www.youtube.com/watch?v=C0DPdy98e4c'


def test_stream():
    v = vipy.video.Video(mp4file)
    for (k,im) in enumerate(v.stream(queuesize=8)):
        if k == 0:
            break
    assert im.shape() == v.shape()
    print('[test_video.video]: stream read  PASSED')
        
    try:
        with v.stream(queuesize=8) as s:
            im = v.preview()
            s.write(im)
        raise Failed()        
    except Failed:
        raise
    except:
        print('[test_video.video]: stream overwrite   PASSED')

    w = vipy.video.Video(tempMP4())
    with w.stream(write=True) as s:
        im = v.preview()
        s.write(im)
    assert w.hasfilename() 
    print('[test_video.video]: stream write   PASSED')       

    v = vipy.video.Video(mp4file)
    for (k,vb) in enumerate(v.stream(queuesize=8).batch(8)):
        if k == 0:
            break
    for (k,vc) in enumerate(v.stream(queuesize=8).clip(8,1)):
        if k == 0:
            break
    assert np.allclose(vb.array(), vc.array())

    for (k,vb) in enumerate(v.stream(queuesize=8).batch(8)):
        if k == 1:
            break
    for (k,vc) in enumerate(v.stream(queuesize=8).clip(8,1)):
        if k == 8:
            break
    assert np.allclose(vb.array(), vc.array())
    
    print('[test_video.video]: stream batch   PASSED')
    print('[test_video.video]: stream clip  PASSED')               
    
    for (k,vb) in enumerate(v.stream(queuesize=8).batch(8)):
        if k == 1:
            break
    for (k,vc) in enumerate(v.load().stream(queuesize=8).clip(8,1)):
        if k == 8:
            break

    assert np.allclose(vb.array(), vc.array(), atol=15)
    print('[test_video.video]: stream clip (loaded) PASSED')                   
    

    # Buffered stream
    for (im1,im2) in zip(v.stream(buffered=True, rebuffered=True), v.stream(buffered=True)):
        break
    assert np.allclose(im1.array(), im2.array(), atol=15)
    
    for (k,(im1,im2,vb)) in enumerate(zip(v.stream(rebuffered=True), v.stream(buffered=True), v.stream(buffered=True).clip(3,1))):
        if k>3:
            break

    assert np.allclose(im1.array(), vb[0].array(), atol=15)
    print('[test_video.video]: buffered stream PASSED')                   
    
    
def _test_video():
    # Common Parameters

    # This fails on ubuntu 20.01, python-3.9 not sure why
    import sys
    if sys.version_info.major == 3 and sys.version_info.minor < 9:    
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
    v.saveas(ignoreErrors=True)
    print('[test_video.video]: saveas(ignoreErrors=True)  PASSED')        

    
    # Clone
    v = vipy.video.RandomVideo(64,64,64)
    v.attributes['clonetest'] = True
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
    vc = v.clone(shallow=True)
    vc.attributes['clonetest'] = False
    assert vc.attributes['clonetest'] != v.attributes['clonetest']

    v = vipy.video.RandomScene(64,64,32)
    vc = v.clone()
    vc.activitymap(lambda a: a.category('clonetest'))
    assert all([a.category() != ac.category() for (a,ac) in zip(v.activitylist(), vc.activitylist())])

    vc = v.clone(shallow=True)
    vc.activitymap(lambda a: a.category('clonetest'))
    assert all([a.category() == ac.category() for (a,ac) in zip(v.activitylist(), vc.activitylist())])
    
    print('[test_video.scene]: clone()  PASSED')        

    # JSON
    v = vipy.video.Video(url=mp4url)    
    vs = vipy.video.Video.from_json(v.clone().json())
    assert v.__dict__ == vs.__dict__
    print('[test_video.scene]: json serialization PASSED')

    # Casting
    v_down = vipy.video.Video.cast(vipy.video.RandomScene(64,64,32))
    v_up = vipy.video.Scene.cast(v_down)
    print('[test_video.scene]: video casting PASSED')    

    # Store/unstore/restore
    v = vipy.video.Video(filename=mp4file).clip(0,30)
    assert v.store().hasattribute('__video__')
    assert np.allclose(v.clone(sanitize=False).restore(tempMP4()).thumbnail(frame=0).load(), v.thumbnail(frame=0).load())
    assert not v.unstore().hasattribute('__video__')
    print('[test_video]: store/unstore/restore PASSED')
    
    
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
    print('[test_video.scene]: activityclip()  PASSED')    
    
    # Activitycrop
    v = vipy.video.RandomSceneActivity(64,64,64)
    vc = v.clone(flushforward=True).filename('Video.mp4')
    assert vc._array is None and v._array is not None
    a = vc.activitycuboid(bb=vipy.geometry.BoundingBox(1,1,width=33, height=33))
    print('[test_video.scene]: activitycuboid()  PASSED - basic only')        
    a = vc.activitytube()
    print('[test_video.scene]: activitytube()  PASSED - basic only')        
    a = vc.actortube(v.tracklist()[0].id())
    print('[test_video.scene]: actortube()  PASSED - basic only')        
    
    # Downloader
    v = vipy.video.Video(url='http://visym.com/out.mp4').load(ignoreErrors=True)
    print('[test_video.video]: download ignoreErrors  PASSED')
    
    #v = vipy.video.Scene(url=mp4url).clip(0,100).load()
    #print('[test_video.scene: download  PASSED')        
    #for im in v:
    #    assert im.shape() == v.shape()
    #print('[test_video.scene]: __iter__  PASSED')
    

    # mp4file in github repo was shrunk, but resize here on the fly to keep old unit tests
    mp4bigger = vipy.video.Video(filename=mp4file).resize(cols=1080, rows=1920).savetmp().filename()  
    vid = vipy.video.Scene(filename=mp4bigger, framerate=30, tracks=[vipy.object.Track(category='person', keyframes=[0,200], boxes=[BoundingBox(xmin=0,ymin=0,width=200,height=400), BoundingBox(xmin=0,ymin=0,width=400,height=100)]),
                                                                     vipy.object.Track(category='vehicle', keyframes=[0,200], boxes=[BoundingBox(xmin=100,ymin=200,width=300,height=400), BoundingBox(xmin=400,ymin=300,width=200,height=100)])])

    # Loader
    v = vid.clone().clip(10,20).load()
    assert len(v) == 10  # FIXME: this is off by one frame for macos sequoia 15.5, homebrew ffmpeg-7.1.1 (clang-1700.0.13.3)
    vc = v.clone(flushforward=True).clip(1,4).load()
    assert len(vc) == 3
    assert all([o1 == o2 for (o1, o2) in zip(vc.frame(1).objects(), vid.frame(12).objects())])
    print('[test_video.scene]:  clip: PASSED')

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

    (H,W) = vid.clone().clip(0,21).load(verbose=False).shape()
    v = vid.clone().clip(0,21).resize(cols=100).load(verbose=False)    
    assert v.width() == 100 and len(v) == 21
    assert np.allclose([im.boundingbox().height() for im in v[0]], [400*(100.0/W), 400*(100.0/W)])
    assert np.allclose([im.boundingbox().width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])
    print('[test_video.scene]: resize isotropic  PASSED')
    
    v = vid.clone().resize(cols=100, rows=100).clip(0,11).load(verbose=False)    
    assert v.width() == 100 and v.height() == 100 and len(v) == 11
    assert np.allclose([im.boundingbox().height() for im in v[0]], [400*(100.0/H), 400*(100.0/H)])
    assert np.allclose([im.boundingbox().width() for im in v[0]], [200*(100.0/W), 300*(100.0/W)])    
    print('[test_video.scene]: resize anisotropic   PASSED')
    
    v = vid.clone().clip(0,22).rot90cw().resize(rows=200).load(verbose=False)
    assert v.height() == 200 and len(v) == 22
    v = v.resize(rows=20, cols=19)
    assert v.height() == 20 and v.width() == 19
    print('[test_video.scene]: rotate and resize  PASSED')
    
    v.annotate('vipy.mp4')
    print('[test_video.scene]: annotate  PASSED')

    v = vid.clone().clip(150,200).rot90cw().resize(rows=200).load(verbose=False)
    assert v.height() == 200 and len(v) == 50
    print('[test_video.scene]: trim, rotate, resize  PASSED')

    v = vid.clone().crop(BoundingBox(xmin=32, ymin=32, width=128, height=128)).clip(0,10).load(verbose=False)
    assert v.height() == 128 
    v = v.crop(BoundingBox(xmin=0, ymin=0, width=10, height=11))  # after load
    assert v.height() == 11 and v.width() == 10 
    print('[test_video.scene]: crop  PASSED')

    v = vid.clone().resize(256,256).randomcrop( (100,200)).clip(0,10).load(verbose=False)
    assert v.height() == 100  and v.width() == 200
    print('[test_video.scene]: randomcrop  PASSED')

    # If the video is not resized, this triggers SIGSEGV on ffmpeg, not sure why
    v = vid.clone().resize(256,256).centercrop( (224,224)).clip(0,10).load(verbose=False)
    assert v.height() == 224  and v.width() == 224  
    print('[test_video.scene]: centercrop  PASSED')

    v = vid.clone().clip(0,30).fliplr().load(verbose=False)
    assert np.allclose(v[0].array(), np.fliplr(vid.clone().clip(0,30).load()[0].array()), atol=15)
    v = v.fliplr()
    assert np.allclose(v[0].array(), vid.clone().clip(0,30).load()[0].array(), atol=15)
    print('[test_video.scene]: fliplr PASSED')  # FIXME: unit test for boxes too

    v = vid.flush().clone().clip(0,30).flipud().load(verbose=False)
    assert np.allclose(v[0].array(), np.flipud(vid.clone().clip(0,30).load()[0].array()))
    print('[test_video.scene]: flipud  PASSED')
    vid.flush()

    v = vid.flush().clone().zeropad(32,64).clip(0,30).load(verbose=False)
    assert v.width() == vid.clone().width()+2*32 and v.height() == vid.clone().height()+2*64 and v.array()[0,0,0,0] == 0 and v.array()[1,-1,-1,-1] == 0
    v = v.zeropad(1,2)  # after load
    assert v.width() == vid.clone().width()+2*32+(1*2) and v.height() == vid.clone().height()+2*64+(2*2) and v.array()[0,0,0,0] == 0 and v.array()[1,-1,-1,-1] == 0
    print('[test_video.scene]: zeropad  PASSED')
    vid.flush()

    v = vid.flush().clone().clip(0,30).resize(256,256).crop(BoundingBox(128,128,width=256,height=257))  
    assert v.width() == 256 and v.height() == 257   
    assert v.load().width() == 256 and v.height() == 257 and v.array()[0,-1,-1,-1] == 0  # load() shape to get the true video size
    print('[test_video.scene]: padcrop PASSED')
    vid.flush()

    v = vid.flush().clone().clip(0,30).resize(rows=128,cols=256).mindim(64)
    assert v.width() == 128 and v.height() == 64   
    assert v.load().width() == 128 and v.height() == 64 
    print('[test_video.scene]: mindim PASSED')
    vid.flush()

    v = vid.flush().clone().clip(0,30).resize(rows=128,cols=256).maxdim(64)
    assert v.width() == 64 and v.height() == 32
    assert v.load().width() == 64 and v.height() == 32 
    print('[test_video.scene]: maxdim PASSED')
    vid.flush()

    v = vid.flush().clone().clip(0,30).resize(rows=127,cols=255).maxsquare()
    assert v.width() == 255 and v.height() == 255
    assert v.load().width() == 255 and v.height() == 255
    print('[test_video.scene]: maxsquare PASSED')
    vid.flush()


    # Thumbnail
    v = vid.clone().thumbnail()
    assert v.width() is not None and v.width() > 0
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

    # Mutable iterator:
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Video(array=frames)
    for im in v.mutable():
        im.numpy()[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Video(array=frames)
    for im in v.mutable().numpy():
        im[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    v = vipy.video.Video(mp4file).clip(0,10).load()
    for im in v.mutable().numpy():
        im[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    v = vipy.video.Video(mp4file).clip(0,10).load()
    for im in v.mutable():
        img = im.numpy()
        img[:,:] = 0
    assert np.sum(v.array().flatten()) == 0
    print('[test_video.scene]: mutable iterator  PASSED')    

    # Scene iterator
    frames = np.random.rand(2,2,2,3).astype(np.float32)
    v = vipy.video.Scene(array=frames, framerate=30)
    for (k,im) in enumerate(v):
        v.add(Detection(0, 0, 0, 100, 100), frame=k)
        try:
            v.add(Track(category=1, keyframes=[1], boxes=[BoundingBox(0,0,1,1)]))
            raise  # framerate is required
        except:
            pass 
        v.add(Track(category=1, keyframes=[1], boxes=[BoundingBox(0,0,1,1)], framerate=30))
        v.add([1,2,3,4], category='test', rangecheck=False, frame=k)
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
    outfile = totempdir('Video.mp4')
    shutil.copy('Video.mp4', outfile)
    v = vipy.video.Video(filename=outfile, startframe=0, endframe=10)
    v2 = v.saveas()
    assert v.hasfilename() and os.path.getsize('Video.mp4') != os.path.getsize(v2.filename()) and os.path.getsize('Video.mp4') == os.path.getsize(v.filename())
    print('[test_video.scene]: saveas()  PASSED')    

    # JSON
    v = vipy.video.RandomScene(64,64,64).flush().url('https://none')
    v._colorspace = None
    vs = vipy.video.Scene.from_json(v.clone().json())
    assert v.pack().__dict__ == vs.pack().__dict__
    print('[test_video.scene]: json serialization PASSED')
    
    # Cleanup
    os.remove(mp4bigger)  
    
def test_clip():
    imgframes = np.zeros( (120,112,112,3), dtype=np.uint8)
    imgframes[60] = imgframes[60]+255
    v = vipy.video.Video(frames=[vipy.image.Image(array=img) for img in imgframes])

    assert np.mean(v[59].array().flatten()) < 128    
    assert np.mean(v[60].array().flatten()) > 128
    assert np.mean(v[61].array().flatten()) < 128    

    vc = v.clip(30, 120)
    assert np.mean(vc[59-30].array().flatten()) < 128    
    assert np.mean(vc[60-30].array().flatten()) > 128
    assert np.mean(vc[61-30].array().flatten()) < 128    

    vc = vc.clip(20, 90)
    assert np.mean(vc[59-50].array().flatten()) < 128    
    assert np.mean(vc[60-50].array().flatten()) > 128
    assert np.mean(vc[61-50].array().flatten()) < 128    

    outfile = totempdir('out.mp4')
    v = vipy.video.Video(frames=[vipy.image.Image(array=img) for img in imgframes])    
    vc = v.saveas(outfile).load()
    assert np.mean(vc.frame(59).array().flatten()) < 128
    assert np.mean(vc.frame(60).array().flatten()) > 128
    assert np.mean(vc.frame(61).array().flatten()) < 128    

    v = vipy.video.Video(frames=[vipy.image.Image(array=img) for img in imgframes], framerate=30)    
    vc = v.saveas(outfile).clip(31, 90).load()  
    assert np.mean(vc.frame(59-31).array().flatten()) < 128
    assert np.mean(vc.frame(60-31).array().flatten()) > 128
    assert np.mean(vc.frame(61-31).array().flatten()) < 128    
    
    v = vipy.video.Video(frames=[vipy.image.Image(array=img) for img in imgframes])    
    vc = v.saveas(outfile).clip(31, 90).clip(2, 40).load()
    assert np.mean(vc.frame(59-31-2).array().flatten()) < 128
    assert np.mean(vc.frame(60-31-2).array().flatten()) > 128
    assert np.mean(vc.frame(61-31-2).array().flatten()) < 128    

    v = vipy.video.RandomScene(64,64,64)
    im = v.frame(10)
    vc = v.saveas(outfile).clip(10,60).load()
    imc = vc.frame(0)
    assert all([i == j for (i,j) in zip(im.objects(), imc.objects())])
    vc = vc.clip(5, 50)

    #assert all([i == j for (i,j) in zip(v.frame(15).objects(), vc.load().frame(0).objects())])
    # FIXME: this is sometimes off by one, why?
        
    os.remove(outfile)    
    print('[test_video.clip]: clip  PASSED')    
    
    
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

    
def test_scene_union():
    
    vid = vipy.video.RandomVideo(64,64,32)
    (rows, cols) = vid.shape()
    track1 = vipy.object.Track(label='Person1', keyframes=[0], boxes=vipy.geometry.BoundingBox(10,20,30,40)).add(2, vipy.geometry.BoundingBox(20,30,40,50))
    track2 = vipy.object.Track(label='Person2', keyframes=1, boxes=[vipy.geometry.BoundingBox(10,20,30,40)]).add(3, vipy.geometry.BoundingBox(30,40,50,60))
    activity = Activity(label='act1', startframe=0, endframe=3).add(track1).add(track2)

    assert track1.boundingbox().ulbr() == (10,20,40,50)
    assert track2.boundingbox().ulbr() == (10,20,50,60)
    #assert activity.boundingbox().ulbr() == vipy.geometry.BoundingBox(10,20,50,60).ulbr()
    #assert activity[0][0].ulbr() == vipy.geometry.BoundingBox(10,20,30,40).ulbr()
    #assert activity[3][0].ulbr() == vipy.geometry.BoundingBox(30,40,50,60).ulbr()                
    #assert activity[1][0].ulbr() == track1[1].ulbr()
    #assert activity[1][1].ulbr() == track2[1].ulbr()
    
    # By reference
    #assert activity.tracks()[track1.id()].category() == 'Person1'
    #track1.category('PersonA')
    #assert activity.tracks()[track1.id()].category() == 'PersonA'    
    #assert 'PersonA' in activity.categories() 
    #track1.category('Person1')
    
    v = vipy.video.Scene(array=vid.array(), colorspace='rgb', category='scene', activities=[activity], tracks=[track1, track2])
    vu = v.clone().union(v)
    assert len(vu.activities())==1

    activity = Activity(label='act2', startframe=0, endframe=3).add(track1).add(track2)
    v2 = vipy.video.Scene(array=v.array(), colorspace='rgb', category='scene', activities=[activity], tracks=[track1, track2])
    vu = v.clone().union(v2)

    assert len(vu.activities())==2
    assert vu.activity_categories() == set(['act2', 'act1'])
    
    track3 = vipy.object.Track(label='Person3', keyframes=[1], boxes=[vipy.geometry.BoundingBox(10,20,30,40)]).add(3, vipy.geometry.BoundingBox(30,40,50,60))    
    activity = Activity(label='act1', startframe=0, endframe=2).add(track1).add(track3)
    v2 = vipy.video.Scene(array=vid.array(), colorspace='rgb', category='scene', activities=[activity])
    vu = v.clone().union(v2)
    assert len(vu.activity_categories()) == 1

    track3 = vipy.object.Track(label='Person3', keyframes=[0], boxes=[vipy.geometry.BoundingBox(10,20,30,40)]).add(2, vipy.geometry.BoundingBox(20,30,40,100))     
    activity = Activity(label='act2', startframe=0, endframe=2).add(track1).add(track3)
    v2 = vipy.video.Scene(array=v.array(), colorspace='rgb', category='scene', activities=[activity])
    vu = v.clone().union(v2)
    
    assert len(vu.activity_categories()) == 2
    print('[test_video.track]: test_scene_union  PASSED')
    


def test_get_frame_meta():
    v = vipy.video.Video(mp4file)
    meta = v.frame_meta()
    frame_types = [m['pict_type'] for m in meta]
    c = Counter(frame_types)
    assert c['B'] == 236
    assert c['P'] == 83
    assert c['I'] == 2

    
if __name__ == "__main__":
    test_stream()
    _test_video()
    _test_scene()
    test_clip()
    test_track()
    test_scene_union()
    test_get_frame_meta()
