import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir
from vipy.geometry import BoundingBox
import pdb

    
def video():    
    # Common Parameters
    mp4url = vipy.videosearch.youtube('owl',1)
    mp4file = '/tmp/out2_1.mp4'
    
    # Empty constructor
    try:
        v = vipy.video.Video()
    except:
        print('Empty constructor: PASSED')


def scene():
    #mp4file = '/tmp/out.mp4'
    mp4file = '/Users/jba3139/Desktop/Video.mp4'    
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

    v = vid.clone().trim(0,10).rescale(0.5).load(verbose=False)
    assert(v.height()*2 == h and v.width()*2 == w)
    print('Video.rescale: PASSED')    

    v = vid.clone().trim(0,200).rot90cw().resize(rows=200).load(verbose=False)
    assert(v.height() == 200)
    print('Video.resize: PASSED')    
    v.annotate('/Users/jba3139/Desktop/vipy.mp4')
    print('Video.annotate: PASSED')

    v = vid.clone().trim(150,200).rot90cw().resize(rows=200).load(verbose=False)
    assert(v.height() == 200)
    v.annotate('/Users/jba3139/Desktop/vipy.mp4')    
    print('Video.resize: PASSED')    


    v = vid.clone().trim(150,200).rot90cw().resize(rows=200).crop(BoundingBox(xmin=0, ymin=0, width=10, height=20)).load(verbose=False)
    assert(v.height() == 20)
    print('Video.crop: PASSED')    
    #v.play('/Users/jba3139/Desktop/vipy.mp4')
    #print('Video.play: PASSED')
    


if __name__ == "__main__":
    video()
    scene()



