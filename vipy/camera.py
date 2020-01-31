import glob
import os
import numpy
import urllib
import timeit
import time
import tempfile
import signal 
import sys
from vipy.util import imresize, tempimage, try_import

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
    def __init__(self, framerate=False, resize=1, grey=False, url=0):

        
        self.CAM = cv2.VideoCapture(url)
        if not self.CAM.isOpened():
            self.CAM.open(url)
        self.FRAMERATE = framerate
        self.RESIZE = resize
        self.GREY = grey
        if framerate:
            self.TIC = timeit.default_timer()

    def __del__(self):
        self.CAM.release()
        return self
        
    def __iter__(self):
        return self

    def _read(self):
        #im = self.CAM.get()
        #return self.CAM.get() # HACK: for slow processing to get most recent image
        return self.CAM.read() # HACK: for slow processing to get most recent image

    def current(self):
        return self.next(grab=False)
                
    def next(self, grab=True):
        k = 0
        while not self.CAM.grab():
            k = k + 1
            #print '[webcam.camera][%d/%d]: invalid grab' % (k, 100)
            if k > 100:
                raise ValueError('Invalid Frame')                
                
        (rval, im) = self.CAM.retrieve()
#        if rval is False:
#            for i in range(0,60):
#                (rval, im) = self._retrieve()
#                if rval is True:
#                    break
        if rval is False:
            raise ValueError('Invalid Frame')
        if self.RESIZE != 1:
            im = imresize(im, self.RESIZE) 
        if self.GREY:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if self.FRAMERATE:
            self.TOC = timeit.default_timer()
            print('[bobo.camera]: frame rate = ' + str(round(1.0/(self.TOC-self.TIC),1)) + ' Hz')
            self.TIC = self.TOC
        return im



class MotionStereo(Webcam):    
    IMCURRENT = None
    IMPREV = None
    
    def next(self):    
        if self.IMPREV is None:
            self.IMPREV = super(MotionStereo, self).next()
        else:
            self.IMPREV = self.IMCURRENT.copy()
        self.IMCURRENT = super(MotionStereo, self).next()
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
        return cv2.imread(self.TMPFILE)  # numpy
  
class VideoCapture(object):
    """ Wraps OpenCV VideoCapture in a generator"""
    def __init__(self, url, do_msec=False):
        """Takes a url, filename, or device id that produces video"""
        self.url = url
        self.do_msec = do_msec
        # Grab video metadata
        self.cap = cv2.VideoCapture(self.url)
        self.num_frames = self.cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv.CV_CAP_PROP_FPS)
        # Release file until we start iteration
        self.cap = self.cap.release()

    def __iter__(self):
        """Yields frames from the video source"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.url)
            self.num_frames = self.cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
            self.fps = self.cap.get(cv.CV_CAP_PROP_FPS)
        ret = True
        try:
            while ret:
                ret, frame = self.cap.read()
                if ret:
                    if self.do_msec:
                        msec = self.cap.get(cv.CV_CAP_PROP_POS_MSEC)
                        yield msec, frame
                    else:
                        yield frame
        finally:
            self.cap = self.cap.release()
