import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir
from vipy.geometry import BoundingBox
import pdb

    
def run():    
    # Common Parameters
    mp4url = vipy.videosearch.youtube(1)
    mp4file = '/tmp/out2_1.mp4'
    
    # Empty constructor
    try:
        v = vipy.video.Video()
    except:
        print('Empty constructor: PASSED')

    # URL

def track():
    #mp4file = '/tmp/out.mp4'
    mp4file = '/Users/jba3139/Desktop/Video.mp4'    
    v = vipy.video.Track(filename=mp4file, track=vipy.object.Track('person', frames=[0,200], boxes=[BoundingBox(0,0,100,100), BoundingBox(100,100,200,200)]))

    v.flush().trim(0,10).load()
    assert len(v) == 10
    v.flush().trim(10,20).load()
    assert len(v) == 10
    print('Video.trim: PASSED')    

    (h,w) = v.shape()
    v.flush().trim(0,10).rot90ccw().load()
    assert v.width() == h and v.height() == w
    print('Video.rot90ccw: PASSED')        

    v.flush().trim(0,10).rot90cw().load()
    assert v.width() == h and v.height() == w    
    print('Video.rot90cw: PASSED')        

    v.flush().trim(0,10).rescale(0.5).load(verbose=False)
    assert(v.height()*2 == h and v.width()*2 == w)
    print('Video.rescale: PASSED')    

    v.flush().trim(0,10).rot90cw().resize(rows=100).load(verbose=False)
    assert(v.height() == 100)
    print('Video.resize: PASSED')    

    v.flush().trim(0,200).rot90ccw().resize(rows=300).load(verbose=False)    
    v.show()
    print('Video.show: PASSED')

    
    #v.show()
    #print(v)

    


if __name__ == "__main__":
    #run()
    track()



