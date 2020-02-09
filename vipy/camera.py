import urllib
import timeit
from vipy.util import tempimage, try_import
from vipy.image import Image


class Camera(object):
    CAM = None
    FRAMERATE = False
    TIC = 0
    TOC = 0
    RESIZE = None
    GREY = None
    PROCESS = None


class Webcam(Camera):
    def __init__(self, framerate=False, url=0):

        try_import("cv2", "opencv-python")
        import cv2

        self.CAM = cv2.VideoCapture(url)
        if not self.CAM.isOpened():
            self.CAM.open(url)
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
        return self.next(grab=False)

    def next(self, grab=True):
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


class Flow(Webcam):
    IMCURRENT = None
    IMPREV = None

    def next(self):
        if self.IMPREV is None:
            self.IMPREV = super(Flow, self).next()
        else:
            self.IMPREV = self.IMCURRENT.copy()
        self.IMCURRENT = super(Flow, self).next()
        return (self.IMCURRENT, self.IMPREV)


class Ipcam(Camera):
    TMPFILE = None

    def __init__(self, url, imfile=tempimage()):
        self.CAM = url
        self.TMPFILE = imfile

    def __iter__(self):
        return self

    def next(self):
        urllib.urlretrieve(self.CAM, self.TMPFILE)
        return Image(array=cv2.imread(self.TMPFILE), colorspace='bgr')
