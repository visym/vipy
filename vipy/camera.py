import urllib
import timeit
from vipy.util import tempimage, try_import, isurl
from vipy.image import Image
from vipy.globals import print

try_import("cv2", "opencv-python")
import cv2


class Camera(object):
    CAM = None
    FRAMERATE = False
    TIC = 0
    TOC = 0
    RESIZE = None
    GREY = None
    PROCESS = None


class Webcam(Camera):
    """Create a webcam object that will yield `vipy.image.Image` frames.

    This is a light wrapper to OpenCV webcam object (cv2.VideoCapture) that yields vipy objects.

    >>> cam = vipy.cmaera.Webcam()
    >>> cam.frame().show()

    Or as an iterator:
    
    >>> for im in vipy.camera.Webcam():
    >>>     im.show()

    To capture a video:

    >>> 
    Args:
        framerate: [float] The framerate to grab from the camera
        url: [int]  The camera index to open 

    """
    def __init__(self, framerate=False, idx=0):

        assert idx >= 0
        self.CAM = cv2.VideoCapture(idx)
        if not self.CAM.isOpened():
            self.CAM.open(idx)
        self.FRAMERATE = framerate
        if framerate:
            self.TIC = timeit.default_timer()

    def __del__(self):
        self.CAM.release()
        return self

    def __iter__(self):
        while(1):
            im = self.next()
            yield im

    def _read(self):
        return self.CAM.read()  # HACK: for slow processing to get most recent image

    def current(self):
        """Alias for `vipy.camera.Webcam.next`"""
        return self.next()

    def next(self):
        """Return a `vipy.image.Image` from the camera"""
        k = 0
        while not self.CAM.grab():
            k = k + 1
            # print '[webcam.camera][%d/%d]: invalid grab' % (k, 100)
            if k > 100:
                raise ValueError('Invalid Frame')

        (rval, im) = self.CAM.retrieve()
        if rval is False:
            raise ValueError('Invalid Frame')
        if self.FRAMERATE:
            self.TOC = timeit.default_timer()
            print('[vipy.camera]: frame rate = ' + str(round(1.0 / (self.TOC - self.TIC),1)) + ' Hz')
            self.TIC = self.TOC

        return Image(array=im, colorspace='bgr')

    def frame(self):
        """Alias for `vipy.camera.Webcam.next`"""
        return self.next()

    def video(self, n, framerate=30):
        """Return a `vipy.video.Video` with n frames, constructed using the provided framerate (defaults to 30Hz)"""
        assert n > 0

        frames = []
        for (k,im) in enumerate(self):
            frames.append(im.rgb())
            if k > n:
                break
        return vipy.video.Video(frames=frames, framerate=framerate)


class Ipcam(Camera):
    """Create a IPcam object that will yield `vipy.image.Image` frames.

    >>> cam = vipy.cmaera.IPcam()
    >>> cam.frame().show()

    Or as an iterator:
    
    >>> for im in vipy.camera.IPcam():
    >>>     im.show()

    """

    TMPFILE = None

    def __init__(self, url, imfile=tempimage()):
        self.CAM = url
        self.TMPFILE = imfile
        assert isurl(url)

    def __iter__(self):
        return self

    def next(self):
        urllib.urlretrieve(self.CAM, self.TMPFILE)
        return Image(array=cv2.imread(self.TMPFILE), colorspace='bgr')
