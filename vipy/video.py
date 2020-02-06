import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase, tempMP4, isurl, isvideourl, templike, tempjpg
from vipy.image import Image, ImageCategory, ImageDetection
from vipy.show import savefig, figure
import vipy.image
import vipy.downloader
import copy
import numpy as np
import ffmpeg
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib
import io
import matplotlib.pyplot as plt
import PIL.Image


class Video(object):
    def __init__(self, url=None, filename=None, framerate=30, rot90cw=False, rot90ccw=False, attributes=None, width=None, height=None):
        self._ignoreErrors = False
        self._url = url
        self._filename = filename
        self._framerate = framerate
        self._array = None
        self.attributes = attributes if attributes is not None else {}
        self._startframe = None
        self._endframe = None
        
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
        assert filename is not None or url is not None, 'Invalid constructor'

        if self._filename is None:
            if 'VIPY_CACHE' in os.environ:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._url))
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            else:
                self._filename = tempMP4()  # guess MP4 for URLs with no file extension
        
        self._ffmpeg = ffmpeg.input(self.filename(), r=framerate)
        if rot90cw:
            self.rot90cw()
        if rot90ccw:
            self.rot90ccw()
        
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        return str('<vipy.video: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames"""
        return len(self._array) if self.isloaded() else 0

    def __getitem__(self, k):
        if k >= 0 and k < len(self):
            return self._array[k]
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        """Streaming video access for large videos that will not fit into memory"""
        # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
        # FIXME: https://github.com/kkroening/ffmpeg-python/issues/78
        self.load()
        for im in self._array[self._startframe:]:
            yield im

    def url(self, url=None, username=None, password=None, sha1=None, ignoreUrlErrors=None):
        """Image URL and URL download properties"""
        if url is None and username is None and password is None:
            return self._url
        if url is not None:
            # set filename and return object
            self.flush()
            self._filename = None
            self._url = url
        if username is not None:
            self._urluser = username  # basic authentication
        if password is not None:
            self._urlpassword = password  # basic authentication
        if sha1 is not None:
            self._urlsha1 = sha1  # file integrity
        if ignoreUrlErrors is not None:
            self._ignoreErrors = ignoreUrlErrors
        return self
            
    def isloaded(self):
        return self._array is not None

    def hasfilename(self):
        return self._filename is not None and os.path.exists(self._filename)

    def hasurl(self):
        return self._url is not None and isurl(self._url)    

    def array(self):
        return self._array
        
    def tonumpy(self):
        return self._array

    def numpy(self):
        return self._array

    def flush(self):
        self._array = None 
        self._ffmpeg = ffmpeg.input(self.filename(), r=self._framerate)
        return self

    def reload(self):
        return self.flush().load()

    def filename(self, newfile=None):
        """Video Filename"""
        if newfile is None:
            return self._filename
        else:
            # set filename and return object
            self.flush()
            self._filename = newfile
            self._url = None
            return self

    def download(self, ignoreErrors=False, timeout=10, verbose=False):
        """Download URL to filename provided by constructor, or to temp filename"""
        if self._url is None and self._filename is not None:
            return self
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[vipy.video.download][ERROR]: Invalid URL "%s" ' % self._url)        

        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if url_scheme in ['http', 'https']:
                vipy.downloader.download(self._url,
                                         self._filename,
                                         verbose=verbose,
                                         timeout=timeout,
                                         sha1=None,
                                         username=None,
                                         password=None)
            elif url_scheme == 'file':
                shutil.copyfile(self._url, self._filename)
            else:
                raise NotImplementedError(
                    'Invalid URL scheme "%s" for URL "%s"' %
                    (url_scheme, self._url))

        except (httplib.BadStatusLine,
                urllib.error.URLError,
                urllib.error.HTTPError):
            if self._ignoreErrors or ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: download failed - Ignoring Video')
                self._array = None
            else:
                raise

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: IO error - Invalid video file, url or invalid write permissions "%s" - Ignoring video' % self.filename())
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: load error for video "%s"' % self.filename())
            else:
                raise
            
        self.flush()
        return self

    def shape(self):
        """Return (height, width) of the frames once they are loaded through the filters"""
        if not self.isloaded():
            raise ValueError('Cannot introspect shape until the file is loaded')
        return (self._array.shape[1], self._array.shape[2])  

    def width(self):
        return self.shape()[1]

    def height(self):
        return self.shape()[0]

    def thumbnail(self, outfile=None, verbose=False):
        """First frame of filtered video, saved to temp file, return vipy.image.Image object"""
        im = Image(filename=tempjpg() if outfile is None else outfile)
        (out, err) = self._ffmpeg.output(im.filename(), vframes=1)\
                                 .overwrite_output()\
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True, capture_stderr=True)
        return im
        
    def load(self, verbose=False):
        """Load a video using ffmpeg, applying the requested filter chain"""
        if self.isloaded():
            return self
        
        # Generate single frame thumbnail to get frame sizes
        (height, width) = self.thumbnail().shape()
            
        (out, err) = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return self


    def clip(self, startframe, endframe):
        """Load a video clip betweeen start and end frames"""
        assert startframe < endframe and startframe >= 0, "Invalid start and end frames" 
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"               
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe) \
                                   .setpts ('PTS-STARTPTS')
        return self
    
    def trim(self, startframe, endframe):
        """Alias for clip"""
        assert startframe < endframe and startframe >= 0, "Invalid start and end frames" 
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"               
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe) \
                                   .setpts ('PTS-STARTPTS')
        return self

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('transpose', 1)
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('transpose', 2)        
        return self
    
    def rescale(self, s):
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"        
        self._ffmpeg = self._ffmpeg.filter('scale', 'iw*%1.2f' % s, 'ih*%1.2f' % s)
        return self
        
    def resize(self, rows=None, cols=None):
        if rows is None and cols is None:
            return self 
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"               
        self._ffmpeg = self._ffmpeg.filter('scale', cols if cols is not None else -1, rows if rows is not None else -1)
        return self

    def framerate(self, fps):
        assert not self.isloaded(), "Filters can only be applied prior to loading, flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
        return self

    def crop(self, bb):
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        self._ffmpeg = self._ffmpeg.crop(bb.xmin(), bb.ymin(), bb.width(), bb.height())
        return self
        
    def saveas(self, outfile, framerate=30, vcodec='libx264'):
        """Save numpy buffer in self._array to a video"""
        assert self.isloaded(), "Video must be loaded prior to saveas"
        (n, height, width, channels) = self._array.shape
        process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
                        .filter('scale', -2, height) \
                        .output(outfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                        .overwrite_output() \
                        .global_args('-loglevel', 'error') \
                        .run_async(pipe_stdin=True)
        
        for frame in self._array:
            process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()
        return outfile


    def greyscale(self):
        pass

    def rgb(self):
        pass
    
    def pptx(self, outfile):
        pass
    
    def show(self):
        f = self.saveas(tempMP4())
        os.system("ffplay %s" % f)

    def torch(self):
        pass

    def clone(self):
        return copy.deepcopy(self)    



class Scene(Video):
    def __init__(self, url=None, filename=None, framerate=30, rot90cw=False, rot90ccw=False, attributes=None, tracks=None, activities=None):
        super(Scene, self).__init__(url=url, filename=filename, framerate=framerate, rot90cw=rot90cw, rot90ccw=rot90ccw, attributes=attributes)

        if tracks is not None:
            assert isinstance(tracks, list) and all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid input"            
        self._tracks = tracks

        if activities is not None:
            assert isinstance(activities, list) and all([isinstance(a, vipy.activity.Activity) for a in activities]), "Invalid input"
        self._activities = activities
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))        
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        if self._tracks is not None:
            strlist.append('tracks=%d' % len(self._tracks))
        if self._activities is not None:
            strlist.append('activities=%d' % len(self._activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        self.load()
        if k >= 0 and k < len(self):
            return vipy.image.Scene(array=self._array[k], colorspace='rgb', objects=[t[k] for t in self._tracks])
        else:
            raise ValueError('Invalid frame index %d ' % k)
    
    def __iter__(self):
        self.load()
        for k in range(0, len(self)):
            yield self.__getitem__(k)

    def trim(self, startframe, endframe):
        super(Scene, self).trim(startframe, endframe)
        self._tracks = [t.offset(dt=-startframe) for t in self._tracks]        
        return self

    def clip(self, startframe, endframe):
        """Alias for trim"""
        super(Scene, self).trim(startframe, endframe)
        self._tracks = [t.offset(dt=-startframe) for t in self._tracks]        
        return self
    
    def crop(self, bb):
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        super(Scene, self).crop(bb)
        self._tracks = [t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for t in self._tracks]                
        return self
        
    def rot90ccw(self):
        (H,W) = self.thumbnail().load().shape()  # yuck, need to get image dimensions before filter        
        self._tracks = [t.rot90ccw(H,W) for t in self._tracks]
        super(Scene, self).rot90ccw()        
        return self

    def rot90cw(self):
        (H,W) = self.thumbnail().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = [t.rot90cw(H,W) for t in self._tracks]
        super(Scene, self).rot90cw()        
        return self

    def resize(self, rows=None, cols=None):
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self.thumbnail().load().shape()  # yuck, need to get image dimensions before filter
        sx = rows/float(H) if rows is not None else cols/float(W)
        sy = cols/float(W) if cols is not None else rows/float(H)        
        self._tracks = [t.scalex(sx) for t in self._tracks]
        self._tracks = [t.scaley(sy) for t in self._tracks]                
        super(Scene, self).resize(rows, cols)
        return self

    def rescale(self, s):
        (H,W) = self.thumbnail().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = [t.rescale(s) for t in self._tracks]
        super(Scene, self).rescale(s)
        return self    
        
    def annotate(self, outfile):
        """Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, save as outfile"""
        assert self.isloaded(), "load() before annotate()"
        vid = self.load().clone()  # to save a new array        
        vid._array = []
        (W, H) = (None, None)
        plt.close(1)
        for (k,im) in enumerate(self.__iter__()):
            imh = im.show(figure=1, nowindow=True)  # sets figure dimensions, does not display window
            if W is None or H is None:
                (W,H) = plt.figure(1).canvas.get_width_height()  # fast
            buf = io.BytesIO()
            plt.figure(1).canvas.print_raw(buf)  # fast
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape( (H, W, 4))
            vid._array.append(np.array(PIL.Image.fromarray(img).convert('RGB')))
        plt.close(1)
        
        vid._array = np.array(vid._array)
        return vid.saveas(outfile)
