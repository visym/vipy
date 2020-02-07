import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir
from vipy.geometry import BoundingBox
import pdb
from vipy.dataset.kinetics import Kinetics400
from vipy.dataset.activitynet import ActivityNet
from vipy.dataset.lfw import LFW


def datasets():
    d = Kinetics400('/tmp/kinetics').download().trainset()
    v = d[0].load()[0].resize(rows=256).saveas('kinetics.jpg')
    d = ActivityNet('/tmp/activitynet').download().dataset()
    v = d[0].load()[0].saveas('activitynet.jpg')
    d = LFW('/tmp/lfw').dataset()
    d[0].saveas('lfw.jpg')
    print('Video.datasets: PASSED')
    
def video():    
    # Common Parameters
    mp4url = vipy.videosearch.youtube('owl',1)
    
    # Empty constructor
    try:
        v = vipy.video.Video()
    except:
        print('Empty constructor: PASSED')

def scene():
    mp4file = 'Video.mp4'    
    vid = vipy.video.Scene(filename=mp4file, tracks=[vipy.object.Track('person', frames=[0,200], boxes=[BoundingBox(xmin=0,ymin=0,width=200,height=400), BoundingBox(xmin=0,ymin=0,width=400,height=100)]),
                                                     vipy.object.Track('vehicle', frames=[0,200], boxes=[BoundingBox(xmin=100,ymin=200,width=300,height=400), BoundingBox(xmin=400,ymin=300,width=200,height=100)])])    

    v = vid.clone().trim(0,10).load()
    assert len(v) == 10
    v.flush().trim(10,20).load()
    assert len(v) == 10
    print('Video.trim: PASSED')    

    (h,w) = v.shape()
    v = vid.clone().trim(0,10).rot90ccw().load()
    assert v.width() == h and v.height() == w
    print('Video.rot90ccw: PASSED')        

    v = vid.clone().trim(0,10).rot90cw().load()
    assert v.width() == h and v.height() == w
    print('Video.rot90cw: PASSED')        

    v = vid.clone().trim(0,10).rescale(0.5).load(verbosity=0)
    assert(v.height()*2 == h and v.width()*2 == w)
    print('Video.rescale: PASSED')    

    v = vid.clone().trim(0,200).rot90cw().resize(rows=200).load(verbosity=0)
    assert(v.height() == 200)
    print('Video.resize: PASSED')    
    v.annotate('vipy.mp4')
    print('Video.annotate: PASSED')

    v = vid.clone().trim(150,200).rot90cw().resize(rows=200).load(verbosity=0)
    assert(v.height() == 200)
    v.annotate('vipy.mp4')    
    print('Video.resize: PASSED')    


    v = vid.clone().trim(150,200).rot90cw().resize(rows=200).crop(BoundingBox(xmin=0, ymin=0, width=10, height=20)).load(verbosity=0)
    assert(v.height() == 20)
    print('Video.crop: PASSED')
    


if __name__ == "__main__":
    datasets()    
    video()
    scene()




