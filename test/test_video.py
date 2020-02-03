import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir
from vipy.geometry import BoundingBox

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
    mp4file = '/tmp/out2_1.mp4'
    video = vipy.video.Video(filename=mp4file).clip(startframe=0, endframe=200)
    v = vipy.video.Track(video=video, tracks=vipy.object.Track('person', frames=[0,200], boxes=[BoundingBox(0,0,100,100), BoundingBox(100,100,200,200)]))
    v.show()
    print(v)

    


if __name__ == "__main__":
    #run()
    track()



