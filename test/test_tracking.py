from pycollector.admin.video import Video
from pycollector.admin.globals import backend
import pycollector.detection
import vipy

def _test_tracking():

    backend('prod','v1')
    v = Video('6C91C543-3930-4D2B-B259-55E19376E84F-3321-0000030A3BFFD8D1')
    
    f = pycollector.detection.ObjectDetector()

    vt = v.clone().load()
    for (k,im) in enumerate(v.stream()):
        imd = f(im)
        vt = vt.assign(k, imd.objects(), miniou=0.2)
        vt.frame(k).show(mutator=vipy.image.mutator_show_trackid(), timestamp='Frame: %d' % k)
        imf = vt.frame(k)
        
        
if __name__ == '__main__':
    _test_tracking()

    
