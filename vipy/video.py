import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase
from vipy.image import Image, ImageCategory, ImageDetection
import copy
import numpy as np

class Video(object):
    """Requires FFMPEG on the path for video files"""
    def __init__(self, frames=None, filenames=None, frameDirectory=None, attributes=None, startframe=0, videofile=None, rot90clockwise=False, rot270clockwise=False, framerate=2):
        if frames is not None:
            self._framelist = frames
        elif videofile is not None:
            imdir = remkdir(os.path.join(filepath(videofile), '.'+filebase(videofile)), flush=True)
            impattern = os.path.join(imdir, '%08d.png')
            if rot90clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=1, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 90 degrees clockwise
            elif rot270clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=2, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 270 degrees clockwise
            else:
                cmd = 'ffmpeg -i "%s" -vf fps=%d -f image2 "%s"' % (videofile, framerate, impattern)
            print('[VideoCategory]: Exporting frames using "%s" ' % cmd)
            os.system(cmd)  # HACK
            self._framelist = [Image(filename=imfile) for imfile in imlist(imdir)]
            self._videofile = videofile            
        elif filenames is not None:
            self._framelist = [Image(filename=imfile) for imfile in filenames]
        elif frameDirectory is not None:
            if not os.path.isdir(frameDirectory):
                raise ValueError('Invalid frames directory "%s"' % frameDirectory)
            self._framelist = [Image(filename=imfile) for imfile in imlist(frameDirectory)]
        else:
            self._framelist = []
            
        self._startframe = startframe

        # Public
        self.attributes = attributes
        
    def __repr__(self):
        return str('<strpy.video: frames=%d>' % (len(self._framelist)))
    
    def __iter__(self):
        for im in self._framelist[self._startframe:]:
            yield im
    
    def __len__(self):
        return len(self._framelist)
            
    def __getitem__(self, k):
        if k >= 0 and k < len(self._framelist) and len(self._framelist) > 0:
            return self._framelist[k]
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def mediatype(self):
        return 'video'

    def append(self, im):
        self._framelist.append(im)
        return self
    
    def frames(self, newframes=None):
        if newframes is None:
            return self._framelist
        else:
            self._framelist = newframes
            return self

    def setattribute(self, key, value):
        if self.attributes is None:
            self.attributes = {key:value}
        else:
            self.attributes[key] = value
        return self

    def show(self, figure=1, colormap=None, do_flush=True):
        for im in self:  # use iterator
            im.show(figure=figure, colormap=colormap)
            #if do_flush:
            #    im.flush()  # flush after showing to avoid running out of memory
        return self
    
    def play(self, figure=1, colormap=None):
        return self.show(figure, colormap)
            
    def map(self, f):
        self._framelist = [f(im) for im in self._framelist]
        return self

    def filter(self, f):
        self._framelist = [im for im in self._framelist if f(im)]
        return self
    
    def clone(self):
        return copy.deepcopy(self)

    def flush(self):
        self._framelist = [im.flush() for im in self._framelist]
        return self

    def isvalid(self):
        return np.all([im.isvalid() for im in self._framelist])

    
class VideoCategory(Video):    
    def __init__(self, frames=None, filenames=None, frameDirectory=None, attributes=None, startframe=0, category=None, videofile=None, rot90clockwise=False, rot270clockwise=False, framerate=2):
        if frames is not None and len(frames) > 0:
            self._framelist = frames
            self._category = frames[0].category()
        elif videofile is not None:
            imdir = remkdir(os.path.join(filepath(videofile), '.'+filebase(videofile)), flush=True)
            impattern = os.path.join(imdir, '%08d.png')
            if rot90clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=1, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 90 degrees clockwise
            elif rot270clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=2, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 270 degrees clockwise
            else:
                cmd = 'ffmpeg -i "%s" -vf fps=%d -f image2 "%s"' % (videofile, framerate, impattern)
            print('[VideoCategory]: Exporting frames using "%s" ' % cmd)
            os.system(cmd)  # HACK
            self._framelist = [ImageCategory(filename=imfile, category=category) for imfile in imlist(imdir)]
            self._videofile = videofile
        elif filenames is not None and len(filenames) > 0:
            self._framelist = [ImageCategory(filename=imfile, category=category) for imfile in filenames]
        elif frameDirectory is not None:
            if not os.path.isdir(frameDirectory):
                raise ValueError('Invalid frames directory "%s"' % frameDirectory)
            self._framelist = [ImageCategory(filename=imfile, category=category) for imfile in imlist(frameDirectory)]
        else:
            self._framelist = []
            
        self._startframe = startframe

        if category is not None:
            self.category(category)
        
        # Public
        self.attributes = attributes
        
    def __repr__(self):
        return str('<strpy.videocategory: frames=%d, category="%s">' % (len(self), self._category))
                    
    def __eq__(self, other):
        return self._category.lower() == other._category.lower()

    def __ne__(self, other):
        return self._category.lower() != other._category.lower()

    def is_(self, other):
        return self.__eq__(other)

    def is_not(self, other):
        return self.__ne__(other)
            
    def __hash__(self):
        return hash(self._category.lower())                

    def iscategory(self, newcategory):
        return (self._category.lower() == newcategory.lower())

    def ascategory(self, newcategory):
        return self.category(newcategory)
    
    def category(self, newcategory=None):
        if newcategory is None:
            return self._category
        else:
            self._category = newcategory
            self.map(lambda im: im.category(newcategory))
            return self

    
class VideoDetection(VideoCategory):
    def __init__(self, frames=None, filenames=None, frameDirectory=None, attributes=None, startframe=0, category=None, boundingboxes=None, rectlist=None, videofile=None, rot90clockwise=False, rot270clockwise=False, framerate=2):
        # Public
        self.attributes = attributes
        
        if frames is not None and len(frames) > 0:
            self._framelist = frames
            self._category = frames[0].category()
            self.attributes = frames[0].attributes
        elif videofile is not None:
            imdir = remkdir(os.path.join(filepath(videofile), '.'+filebase(videofile)), flush=True)
            impattern = os.path.join(imdir, '%08d.png')
            if rot90clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=1, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 90 degrees clockwise
            elif rot270clockwise:
                cmd = 'ffmpeg -i "%s" -vf "transpose=2, fps=%d" -f image2 "%s"' % (videofile, framerate, impattern)  # 270 degrees clockwise
            else:
                cmd = 'ffmpeg -i "%s" -vf fps=%d -f image2 "%s"' % (videofile, framerate, impattern)
            print('[VideoCategory]: Exporting frames using "%s" ' % cmd)
            os.system(cmd)  # HACK
            self._framelist = [ImageDetection(filename=imfile, category=category) for imfile in imlist(imdir)]
            self._videofile = videofile
            
        elif filenames is not None:
            self._framelist = [ImageDetection(filename=imfile, category=category) for imfile in filenames]
        elif frameDirectory is not None:
            if not os.path.isdir(frameDirectory):
                raise ValueError('Invalid frames directory "%s"' % frameDirectory)
            self._framelist = [ImageDetection(filename=imfile, category=category) for imfile in imlist(frameDirectory)]
            
        if boundingboxes is not None:
            self._framelist = [im.boundingbox(bb) for (im,bb) in zip(self._framelist, boundingboxes)]
        elif rectlist is not None:
            self._framelist = [im.boundingbox(xmin=r[0], ymin=r[1], xmax=r[2], ymax=r[3]) for (im,r) in zip(self._framelist, rectlist)]
            
        self._startframe = startframe

        if category is not None:
            self.category(category)
        
                                
    def __repr__(self):
        return str('<strpy.videodetection: frames=%d, category="%s">' % (len(self), str(self.category())))
    


class VideoCapture(object):
    """ Wraps OpenCV VideoCapture in a generator"""
    import cv2 
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
