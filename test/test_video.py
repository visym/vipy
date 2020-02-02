import os
import numpy as np
from vipy.video import Video
from vipy.util import tempjpg, tempdir
import vipy.videosearch 

def run():    
    # Common Parameters
    mp4url = vipy.videosearch.youtube(1)

    

    # Empty constructor
    try:
        v = Video()
    except:
        print('Empty constructor: PASSED')

    # URL
    


if __name__ == "__main__":
    run()



