import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase
from vipy.image import Image, ImageCategory, ImageDetection
import copy
import numpy as np

#https://github.com/kkroening/ffmpeg-python/issues/246
# put ffmpeg on all cluster machines


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
        return str('<strpy.videoframes: frames=%d>' % (len(self._framelist)))
    
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

    
class VideoCategory(Video):
    pass

class VideoDetection(Video):
    pass
