import vipy


def _test_tracking():

    from pycollector.admin.video import Video
    from pycollector.admin.globals import backend
    import pycollector.detection

    backend('prod','v1')
    v = Video('6C91C543-3930-4D2B-B259-55E19376E84F-3321-0000030A3BFFD8D1')
    
    f = pycollector.detection.ObjectDetector()

    v = v.stabilize(show=True)
    vt = v.clone().load()
    vo = vipy.video.Video('/tmp/out.mp4')
    with vo.stream(write=True, overwrite=True) as s:
        for (k,im) in enumerate(v):
            imd = f(im, conf=2E-1)
            vt = vt.assign(k, imd.objects())
            vt.frame(k).show(mutator=vipy.image.mutator_show_trackid(), timestamp='Frame: %d' % k)
            s.write(vt.frame(k).annotate(timestamp='Frame: %d' % k, mutator=vipy.image.mutator_show_trackid()))
        
        
if __name__ == '__main__':
    _test_tracking()

    
