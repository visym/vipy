import numpy as np
from vipy.geometry import BoundingBox


def run():
    try:
        bb = BoundingBox()
        raise
    except:
        print('Empty constructor: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=10, height=10.2)
    print('(x,y,w,h) constructor: PASSED')        

    bb = BoundingBox(xmin='0', ymin='0.5', width='1E2', height='10.2')
    print('(x,y,w,h) string constructor: PASSED')        

    bb = BoundingBox(xmin='0', ymin='0.5', xmax='1E2', ymax='1000.2')
    print('(xmin,ymin,xmax,ymax) constructor: PASSED')        

    bb = BoundingBox(centroid=('0',0), width=10, height=10)
    print('(centroid, width, height) constructor: PASSED')        

    try:
        bb = BoundingBox(mask=np.zeros( (10,10) ))        
        raise
    except:
        print('Degenerate mask constructor: PASSED')        
        
    bb = BoundingBox(xmin=0, ymin=0, width=-100, height=0)
    if not bb.isdegenerate():
        raise
    print('Invalid box: PASSED')
        

if __name__ == "__main__":
    run()



