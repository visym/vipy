import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase
from vipy.image import Image, ImageCategory, ImageDetection
import copy
import numpy as np

#https://github.com/kkroening/ffmpeg-python/issues/246
# put ffmpeg on all cluster machines

#need ActivityDetection, ActivityClassification
#raw video with ffmpeg support
#activities are associated with one or more objects and have a start and end time
#a video may have no detections
#a video may only have a classificatiojn (consent video)
#we will want to run face and object detectors on these videos
#we will want to clip videos
#we will want to stabilize videos
#resize videos
#extract single frame at activity
#save videos
#show frames of video
#each frame should have one or more activities and one or more objects
#each activity should have one or more objects
#individual activities should be accessible as a clip
#does a frame contain an activity
#all frames that contain an activity
#we want to track and filter boxes
#export refined json
#read video metadata
#save an overlay video
#preprocess clips for torch
#constructor should not take JSON file input

#an object can have more than one class (identity and face) instance (my face on tuesday), identity (my face), category (face), attribute (adjectives, gender)
#verbs can also have attributes 


class Scene(object):
    def __init__(self, video, activities=None, tracks=None, attributes=None):    
        pass

    def __getitem__(self, k):
        self.load()
        if k >= 0 and k < len(self):
            return vipy.image.Scene()
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def saveas(self, outfile):
        pass

    def activityclip(self, k):
        return Scene()

    
class Video(object):
    def __init__(self, url=None, filename=None, startframe=0, framerate=30, rot90cw=False, rot90ccw=False, attributes=None):
        self._url = url
        self._filename = filename
        self._framerate = framerate
        self._array = None
        
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
        assert filename is not None or url is not None, 'Invalid constructor'
        
        
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], self.attributes['colorspace']))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        return str('<vipy.image: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames"""
        return 1

    def __getitem__(self, k):
        self.load()
        if k >= 0 and k < len(self):
            return self._array[k]
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        self.load()
        for im in self._array[self._startframe:]:
            yield im
        
    
    def load(self):
        # Download video from URL and save to file (define cache)
        # Load file into memory as an array of numpy frames
        self._array = None
        return self

    def clip(self, startframe, endframe):
        self._array = self._array[startframe:endframe]
        # Update activities and objects
        self.activities = [a.offset(startframe) for a in self.activities]
        self.objects = None
        return self

    def saveas(self, outfile):
        pass

    def show(self):
        pass

    def torch(self):
        pass

    def numpy(self):
        return self._array
    
    def resize(self, width, height):
        pass

    
    
class VideoFrames(object):
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
            print('[vipy.video.VideoFrames]: Exporting frames using "%s" ' % cmd)
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
        return str('<vipy.videoframes: frames=%d>' % (len(self._framelist)))
    
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

    def show(self, figure=1, colormap=None, do_flush=False):
        for im in self:  # use iterator
            im.show(figure=figure, colormap=colormap)
            if do_flush:
                im.flush()  # flush after showing to avoid running out of memory
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

    
