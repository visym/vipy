import os
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng, \
    istuple, islist, isnumber
from vipy.image import Image
import vipy.geometry
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
import warnings
import shutil
import types


class Video(object):
    """ vipy.video.Video class

    The vipy.video class provides a fluent, lazy interface for representing, transforming and visualizing videos.
    The following constructors are supported:

    >>> vid = vipy.video.Video(filename='/path/to/video.ext')

    Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm'].

    >>> vid = vipy.video.Video(url='https://www.youtube.com/watch?v=MrIN959JuV8')
    >>> vid = vipy.video.Video(url='http://path/to/video.ext', filename='/path/to/video.ext')

    Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename().  If the environment
    variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory.
    If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename.
    URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' (https://ytdl-org.github.io/youtube-dl/supportedsites.html)

    >>> vid = vipy.video.Video(array=frames, colorspace='rgb')
    
    The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video
    Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load().

    """
    def __init__(self, filename=None, url=None, framerate=None, attributes=None, array=None, colorspace=None):
        self._url = None
        self._filename = None
        self._array = None
        self._colorspace = None
        self._ffmpeg = None
        self._framerate = None
        
        self.attributes = attributes if attributes is not None else {}
        assert filename is not None or url is not None or array is not None, 'Invalid constructor - Requires "filename", "url" or "array"'

        # Input
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
            self.url(url)
        if filename is not None:
            self.filename(filename)
        else:
            if isvideourl(self._url):
                self._filename = templike(self._url)
            elif isyoutubeurl(self._url):
                self._filename = os.path.join(tempdir(), '%s' % self._url.split('?')[1])
            if 'VIPY_CACHE' in os.environ and self._filename is not None:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._filename))
        if array is not None:
            self.array(array)
            self.colorspace(colorspace)
            
        # Video filter chain
        self._ffmpeg = ffmpeg.input(self.filename())        
        if framerate is not None:
            self.framerate(framerate)
            self._framerate = framerate
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, color=%s" % (self.height(), self.width(), len(self), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self._framerate is not None:
            strlist.append('fps=%s' % str(self._framerate))
        return str('<vipy.video: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames"""
        return len(self._array) if self.isloaded() else 0

    def __getitem__(self, k):
        """Return the kth frame as an vipy.image object"""
        if k >= 0 and k < len(self):
            return Image(array=self._array[k], colorspace=self.colorspace())
        elif not self.isloaded():
            raise ValueError('Video not loaded, load() before indexing')
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        """Iterate over frames, yielding vipy.image.Image object for each frame"""
        self.load()
        self._array = np.copy(self._array) if not self._array.flags['WRITEABLE'] else self._array  # triggers copy
        with np.nditer(self._array, op_flags=['readwrite']) as it:
            for k in range(0, len(self)):
                yield self.__getitem__(k)

    def stream(self):
        # FIXME: Streaming video access for large videos that will not fit into memory
        # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
        # FIXME: https://github.com/kkroening/ffmpeg-python/issues/78
        raise NotImplementedError('Streaming video access for large videos that will not fit into memory - Try clip() first')
        
    def __array__(self):
        """Called on np.array(self) for custom array container, (requires numpy >=1.16)"""
        return self.numpy()
                
    def dict(self):
        video = {'filename':self.filename(),
                 'url':self.url(),
                 'ffmpeg':str(self._ffmpeg.output('dummyfile').compile()),
                 'height':self.height() if self.isloaded() else None,
                 'width':self.width() if self.isloaded() else None,
                 'channels':self.channels() if self.isloaded() else None,
                 'colorspace':self.colorspace(),
                 'framerate':self._framerate,
                 'attributes':self.attributes,
                 'array':self.array()}
        return {'video':video}
             
    def take(self, n):
        """Return n frames from the clip uniformly spaced as numpy array"""
        assert self.isloaded(), "Load() is required before take()"""
        dt = int(np.round(len(self._array) / float(n)))  # stride
        return self._array[::dt][0:n]

    def framerate(self, fps):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first"        
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
        self._framerate = fps
        return self
            
    def colorspace(self, colorspace=None):
        """Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']"""
        if colorspace is None:
            return self._colorspace
        elif self.isloaded():
            assert str(colorspace).lower() in ['rgb', 'bgr', 'lum', 'float']
            if self.array().dtype == np.float32:
                assert str(colorspace).lower() in ['float']
            elif self.array().dtype == np.uint8:
                assert str(colorspace).lower() in ['rgb', 'bgr', 'lum']
                if str(colorspace).lower() in ['lum']:
                    assert self.channels() == 1, "Luminance colorspace must be one channel uint8"
                elif str(colorspace).lower() in ['rgb', 'bgr']:
                    assert self.channels() == 3, "RGB or BGR colorspace must be three channel uint8"
            else:
                raise ValueError('Invalid array() type "%s" - only np.float32 or np.uint8 allowed' % str(self.array().dtype))
            self._colorspace = str(colorspace).lower()
        return self

    def url(self, url=None, username=None, password=None, sha1=None):
        """Image URL and URL download properties"""
        if url is None and username is None and password is None:
            return self._url
        if url is not None:
            # set filename and return object
            if self.isloaded():
                self.flush()
            self._filename = None
            self._url = url
        if username is not None:
            self._urluser = username  # basic authentication
        if password is not None:
            self._urlpassword = password  # basic authentication
        if sha1 is not None:
            self._urlsha1 = sha1  # file integrity
        return self

    def isloaded(self):
        """Return True if the video has been loaded"""
        return self._array is not None

    def channels(self):
        """Return integer number of color channels"""
        return 1 if self.load().array().ndim == 3 else self.load().array().shape[3]

    def iscolor(self):
        return self.channels() == 3

    def isgrayscale(self):
        return self.channels() == 1

    def hasfilename(self):
        return self._filename is not None and os.path.exists(self._filename)

    def isdownloaded(self):
        return self._filename is not None and os.path.exists(self._filename)

    def hasurl(self):
        return self._url is not None and isurl(self._url)

    def array(self, array=None, copy=False):
        if array is None:
            return self._array
        elif isnumpy(array):
            assert array.dtype == np.float32 or array.dtype == np.uint8, "Invalid input - array() must be type uint8 or float32"
            assert array.ndim == 4, "Invalid input array() must be of shape NxHxWxC, for N frames, of size HxW with C channels"
            self._array = np.copy(array) if copy else array
            self._array.setflags(write=True)  # mutable iterators
            self.colorspace(None)  # must be set with colorspace() after array() before _convert()
            return self
        else:
            raise ValueError('Invalid input - array() must be numpy array')            

    def fromarray(self, array):
        """Alias for self.array(..., copy=True), which forces the new array to be a copy"""
        return self.array(array, copy=True)
    
    def tonumpy(self):
        """Alias for numpy()"""
        return self.numpy()

    def numpy(self):
        """Convert the video to a numpy array, triggers a load()"""
        self.load()
        self._array = np.copy(self._array) if not self._array.flags['WRITEABLE'] else self._array  # triggers copy 
        return self._array

    def flush(self):
        self._array = None
        self._ffmpeg = ffmpeg.input(self.filename())  # restore, no other filters
        return self

    def reload(self):
        return self.flush().load()

    def filename(self, newfile=None):
        """Video Filename"""
        if newfile is None:
            return self._filename
        else:
            # set filename and return object
            self._filename = newfile
            self.flush()  # update self._ffmpeg object
            return self

    def download(self, ignoreErrors=False, timeout=10, verbose=False):
        """Download URL to filename provided by constructor, or to temp filename"""
        if self._url is None and self._filename is not None:
            return self
        if self._url is None:
            raise ValueError('[vipy.video.download]: No URL to download')
        elif not isurl(str(self._url)):
            raise ValueError('[vipy.video.download]: Invalid URL "%s" ' % self._url)

        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if isyoutubeurl(self._url):
                vipy.videosearch.download(self._url, self._filename, writeurlfile=False, skip=ignoreErrors)
                if not self.hasfilename():
                    for ext in ['mkv', 'mp4', 'webm']:
                        f = '%s.%s' % (self.filename(), ext)
                        if os.path.exists(f):
                            os.symlink(f, self.filename())  # file extension not known until download(), symlink for caching
                            self.filename(f)
                    if not self.hasfilename():
                        raise ValueError('Downloaded file not found "%s.*"' % self.filename())
            elif url_scheme in ['http', 'https']:
                vipy.downloader.download(self._url,
                                         self._filename,
                                         verbose=verbose,
                                         timeout=timeout,
                                         sha1=None,
                                         username=None,
                                         password=None)
            elif url_scheme == 'file':
                shutil.copyfile(self._url, self._filename)
            elif url_scheme == 's3':
                raise NotImplementedError('S3 support is in development')
            else:
                raise NotImplementedError(
                    'Invalid URL scheme "%s" for URL "%s"' %
                    (url_scheme, self._url))

        except (httplib.BadStatusLine,
                urllib.error.URLError,
                urllib.error.HTTPError):
            if ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: download failed - Ignoring Video')
                self._array = None
            else:
                raise

        except IOError:
            if ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: IO error - Invalid video file, url or invalid write permissions "%s" - Ignoring video' % self.filename())
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if ignoreErrors:
                warnings.warn('[vipy.video][WARNING]: load error for video "%s"' % self.filename())
            else:
                raise
        return self

    def shape(self):
        """Return (height, width) of the frames, can only be introspected after load()"""
        if not self.isloaded():
            raise ValueError('Cannot introspect shape until the file is loaded')
        return (self._array.shape[1], self._array.shape[2])

    def width(self):
        """Width (cols) in pixels of the video, can only be introspected after load()"""
        return self.shape()[1]

    def height(self):
        """Height (rows) in pixels of the video, can only be introspected after load()"""        
        return self.shape()[0]

    def _preview(self, outfile=None, verbose=False):
        """Return first frame of filtered video, saved to temp file, return vipy.image.Image object.  This is useful for previewing the frame shape of a complex filter chain."""
        if self.isloaded():
            return self[0]
        elif not self.hasfilename():
            raise ValueError('Video file not found')
        im = Image(filename=tempjpg() if outfile is None else outfile)
        (out, err) = self._ffmpeg.output(im.filename(), vframes=1)\
                                 .overwrite_output()\
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True, capture_stderr=True)
        return im

    def load(self, verbosity=1, ignoreErrors=False, startframe=None, endframe=None, rotation=None, rescale=None, mindim=None):
        """Load a video using ffmpeg, applying the requested filter chain.  
           If verbosity=2. then ffmpeg console output will be displayed. 
           If ignoreErrors=True, then download errors are warned and skipped.
           Filter chains can be included at load time using the following kwargs:
               * (startframe=s, endframe=e) -> self.clip(s, e)
               * rotation='rot90cw' -> self.rot90cw()
               * rotation='rot90ccw' -> self.rot90ccw()        
               * rescale=s -> self.rescale(s)
               * mindim=d -> self.mindim(d)
        """
        if self.isloaded():
            return self
        elif not self.hasfilename() and not self.isloaded():
            self.download(ignoreErrors=ignoreErrors)
        if not self.hasfilename() and ignoreErrors:
            print('[vipy.video.load]: Video file "%s" not found - Ignoring' % self.filename())
            return self
        if verbosity > 0:
            print('[vipy.video.load]: Loading "%s"' % self.filename())

        # Increase filter chain from load() kwargs
        assert (startframe is not None and startframe is not None) or (startframe is None and endframe is None), "(startframe, endframe) must both be provided"
        if startframe is not None and endframe is not None:   
            self.trim(startframe, endframe)  # trim first
        assert not (rescale is not None and mindim is not None), "mindim and rescale cannot both be provided, choose one or the other, or neither"            
        if mindim is not None:
            self.mindim(mindim)   # resize second
        if rescale is not None:
            self.rescale(rescale)      
        if rotation is not None:  
            if rotation == 'rot90cw':
                self.rot90cw()  # rotate third
            elif rotation == 'rot90ccw':
                self.rot90ccw()
            else:
                raise ValueError("rotation must be one of ['rot90ccw', 'rot90cw']")
    
        # Generate single frame _preview to get frame sizes
        imthumb = self._preview(verbose=verbosity > 1)
        (height, width, channels) = (imthumb.height(), imthumb.width(), imthumb.channels())
        (out, err) = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                                 .global_args('-loglevel', 'debug' if verbosity > 1 else 'error') \
                                 .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, channels])  # read-only
        self.colorspace('rgb' if channels == 3 else 'lum')
        return self

    def clip(self, startframe, endframe):
        """Load a video clip betweeen start and end frames"""
        assert startframe <= endframe and startframe >= 0, "Invalid start and end frames (%s, %s)" % (str(startframe), str(endframe))
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first"
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe)\
                                   .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
        return self

    def trim(self, startframe, endframe):
        """Alias for clip"""
        assert startframe <= endframe and startframe >= 0, "Invalid start and end frames"
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe)\
                                   .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
        return self

    def rot90cw(self):
        """Rotate the video 90 degrees clockwise, can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('transpose', 1)
        return self

    def rot90ccw(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('transpose', 2)
        return self

    def rescale(self, s):
        """Rescale the video by factor s, such that the new dimensions are (s*H, s*W), can only be applied prior load()"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('scale', 'iw*%1.2f' % s, 'ih*%1.2f' % s)
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to be (rows, cols), can only be applied prior to load()"""
        if rows is None and cols is None:
            return self
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        self._ffmpeg = self._ffmpeg.filter('scale', cols if cols is not None else -1, rows if rows is not None else -1)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"        
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"        
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def crop(self, bb):
        """Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        self._ffmpeg = self._ffmpeg.crop(bb.xmin(), bb.ymin(), bb.width(), bb.height())
        return self

    def saveas(self, outfile, framerate=30, vcodec='libx264', verbose=False):
        """Save video to new output video file.  This function does not draw boxes, it saves pixels to a new video file.
        If self.array() is loaded, then export the contents of self._array to the video file
        If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video
        """
        if self.isloaded():
            # Save numpy() from load() to video
            (n, height, width, channels) = self._array.shape
            process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
                            .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                            .output(outfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                            .overwrite_output() \
                            .global_args('-loglevel', 'error' if not verbose else 'debug') \
                            .run_async(pipe_stdin=True)

            for frame in self._array:
                process.stdin.write(frame.astype(np.uint8).tobytes())

            process.stdin.close()
            process.wait()
        elif self.isdownloaded():
            # Transcode the video file directly, do not load() then export
            self._ffmpeg.filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                        .output(outfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                        .overwrite_output() \
                        .global_args('-loglevel', 'error' if not verbose else 'debug') \
                        .run()
        elif self.hasurl():
            raise ValueError('Input video url "%s" not downloaded, call download() first' % self.url())
        else:
            raise ValueError('Input video file not found "%s"' % self.filename())

        return outfile

    def pptx(self, outfile):
        """Export the video in a format that can be played by powerpoint"""
        pass

    def play(self, verbose=True):
        """Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it or try saveas(tempMP4())"""
        if not self.isdownloaded():
            self.download()
        if not self.hasfilename():
            if verbose:
                print('[vipy.video.play]: Saving video to temporary file "%s"' % f)            
            f = self.saveas(tempMP4())
        cmd = "ffplay %s" % f
        if verbose:
            print('[vipy.video.play]: Executing "%s"' % cmd)
        os.system(cmd)
        return self

    def show(self):
        """Alias for play()"""
        return self.play()
    
    def torch(self, take=None):
        """Convert the loaded video to an NxCxHxW torch tensor, forces a load()"""
        try_import('torch'); import torch

        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        self.load()
        frames = self._array if self.iscolor() else np.expand_dims(self._array, 3)
        t = torch.from_numpy(frames.transpose(0,3,1,2))
        return t if take is None else t[::int(np.round(len(t)/float(take)))][0:take]

    def clone(self):
        """Copy the video object"""
        return copy.deepcopy(self)

    def map(self, func):
        """Apply lambda function to the loaded numpy array img, changes pixels not shape
        
        Lambda function must have the following signature:
            * newimg = func(img)
            * img: HxWxC numpy array for a single frame of video
            * newimg:  HxWxC modified numpy array for this frame.  Change only the pixels, not the shape

        The lambda function will be applied to every frame in the video in frame index order.
        """
        assert isinstance(func, types.LambdaType), "Input must be lambda function with np.array() input and np.array() output"
        oldimgs = self.load().array()
        self.array(np.apply_along_axis(func, 0, self._array))   # FIXME: in-place operation?
        if (any([oldimg.dtype != newimg.dtype for (oldimg, newimg) in zip(oldimgs, self.array())]) or
            any([oldimg.shape != newimg.shape for (oldimg, newimg) in zip(oldimgs, self.array())])):            
            self.colorspace('float')  # unknown colorspace after shape or type transformation, set generic
        return self

    
class VideoCategory(Video):
    """vipy.video.VideoCategory class
    """
    def __init__(self, filename=None, url=None, framerate=30, attributes=None, category=None, startframe=None, endframe=None, array=None, colorspace=None):
        super(VideoCategory, self).__init__(url=url, filename=filename, framerate=framerate, attributes=attributes, array=array, colorspace=colorspace)
        self._category = category
        if startframe is not None and endframe is not None:
            self._startframe = startframe
            self._endframe = endframe
            self.trim(startframe, endframe)

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if not self.isloaded() and self._startframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        return str('<vipy.video.VideoCategory: %s>' % (', '.join(strlist)))

    def dict(self):
        d = super(VideoCategory, self).dict()
        d['category'] = self.category()
        return d
    
    def category(self, c=None):
        if c is None:
            return self._category
        else:
            self._category = c
            return self
    

class Scene(VideoCategory):
    """ vipy.video.Scene class

    The vipy.video.Scene class provides a fluent, lazy interface for representing, transforming and visualizing annotated videos.
    The following constructors are supported:

    >>> vid = vipy.video.Scene(filename='/path/to/video.ext')

    Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm'].

    >>> vid = vipy.video.Scene(url='https://www.youtube.com/watch?v=MrIN959JuV8')
    >>> vid = vipy.video.Scene(url='http://path/to/video.ext', filename='/path/to/video.ext')

    Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename().  If the environment
    variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory.
    If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename.
    URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' 
    [https://ytdl-org.github.io/youtube-dl/supportedsites.html]

    >>> vid = vipy.video.Scene(array=frames, colorspace='rgb')
    
    The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video
    Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load().

    >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum')
    
    The input 'greyframes' is an NxHxWx1 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video
    This corresponds to the luminance of an RGB colorspace

    >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum', tracks=tracks, activities=activities)

         * tracks = [vipy.object.Track(), ...]
         * activities = [vipy.object.Activity(), ...]
 
    The inputs are lists of tracks and/or activities.  An object is a spatial bounding box with a category label.  A track is a spatiotemporal bounding 
    box with a category label, such that the box contains the same instance of an object.  An activity is one or more tracks with a start and end frame for an 
    activity performed by the object instances.

    """
        
    def __init__(self, filename=None, url=None, framerate=None, array=None, colorspace=None, category=None, tracks=None, activities=None, attributes=None):
        super(Scene, self).__init__(url=url, filename=filename, framerate=None, attributes=attributes, array=array, colorspace=colorspace, category=category)

        self.tracks = {}
        if tracks is not None:
            tracks = tracks if isinstance(tracks, list) or isinstance(tracks, tuple) else [tracks]  # canonicalize
            assert all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid track input; tracks=[vipy.object.Track(), ...]"
            self.tracks = {t.id():t for t in tracks}

        self.activities = {}
        if activities is not None:
            activities = activities if isinstance(activities, list) or isinstance(activities, tuple) else [activities]  # canonicalize            
            assert all([isinstance(a, vipy.object.Activity) for a in activities]), "Invalid activity input; activities=[vipy.object.Activity(), ...]"
            self.activities = {a.id():a for a in activities}

        if framerate is not None:
            self.framerate(framerate)
            
        self._currentframe = None  # used during iteration only
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, color=%s" % (self.height(), self.width(), len(self._array), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self._framerate is not None:
            strlist.append('fps=%s' % str(self._framerate))            
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if len(self.tracks) > 0:
            strlist.append('objects=%d' % len(self.tracks))
        if len(self.activities) > 0:
            strlist.append('activities=%d' % len(self.activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        """Return the vipy.image.Scene() for the vipy.video.Scene() interpolated at frame k"""
        if self.load().isloaded() and k >= 0 and k < len(self):
            dets = [t[k] for (tid,t) in self.tracks.items() if t[k] is not None]  # track interpolation with boundary handling
            for d in dets:
                for (aid, a) in self.activities.items():
                    if a.hastrack(d.attributes['track']) and a.during(k):
                        d.category(a.category())  # Category of detection is activity, 
                        d.shortlabel(a.shortlabel())  # see d.attributes['track'] for original labels
                        if 'activity' not in d.attributes:
                            d.attributes['activity'] = []                            
                        d.attributes['activity'].append(a)  # for activity correspondence
            return vipy.image.Scene(array=self._array[k], colorspace=self.colorspace(), objects=dets, category=self.category())  
        elif not self.isloaded():
            raise ValueError('Video not loaded; load() before indexing')
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        """Iterate over every frame of video yielding interpolated vipy.image.Scene() at the current frame"""
        self.load()
        for k in range(0, len(self)):
            self._currentframe = k            
            yield self.__getitem__(k)
        self._currentframe = None
            
    def add(self, obj, category=None, attributes=None):
        """Add the object obj to the scene, and return an index to this object for future updates
        
        This function is used to incrementally build up a scene frame by frame.  Obj can be one of the following types:

            * obj = vipy.object.Detection(), this must be called from within a frame iterator (e.g. for im in video) to get the current frame index
            * obj = vipy.object.Track()  
            * obj = vipy.object.Activity()
            * obj = [xmin, ymin, width, height], with associated category kwarg, this must be called from within a frame iterator to get the current frame index
        
        It is recomended that the objects are added as follows.  For a scene=vipy.video.Scene():
            
            for im in scene:
                # Do some processing on frame im to detect objects
                (object_labels, xywh) = object_detection(im)

                # Add them to the scene, note that each object instance is independent in each frame, use tracks for object correspondence
                for (lbl,bb) in zip(object_labels, xywh):
                    scene.add(bb, lbl)

                # Do some correspondences to track objects
                t2 = scene.add( vipy.object.Track(...) )

                # Update a previous track to add a keyframe
                scene.track[t2].add( ... )
        
        This will keep track of the current frame in the video and add the objects in the appropriate place

        """
        if isinstance(obj, vipy.object.Detection):
            assert self._currentframe is not None, "add() for vipy.object.Detection() must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=obj.category(), keyframes=[self._currentframe], boxes=[obj], boundary='strict', attributes=obj.attributes)
            self.tracks[t.id()] = t
            return t.id()
        elif isinstance(obj, vipy.object.Track):
            self.tracks[obj.id()] = obj
            return obj.id()
        elif isinstance(obj, vipy.object.Activity):
            self.activities[obj.id()] = obj
            return obj.id()
        elif (istuple(obj) or islist(obj)) and len(obj) == 4 and isnumber(obj[0]):
            assert self._currentframe is not None, "add() for obj=xywh must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=category, keyframes=[self._currentframe], boxes=[vipy.geometry.BoundingBox(xywh=obj)], boundary='strict', attributes=attributes)
            self.tracks[t.id()] = t
            return t.id()
        else:
            raise ValueError('Undefined object type "%s" to be added to scene - Supported types are obj in ["vipy.object.Detection", "vipy.object.Track", "vipy.object.Activity", "[xmin, ymin, width, height]"]' % str(type(obj)))        
        
    def dict(self):
        d = super(Scene, self).dict()
        d['category'] = self.category()
        d['tracks'] = [t.dict() for t in self.tracks.values()]
        d['activities'] = [a.dict() for a in self.activities.values()]
        return d
        
    def framerate(self, fps):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first"        
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
        self.tracks = {k:t.framerate(fps) for (k,t) in self.tracks.items()}
        self.activities = {k:a.framerate(fps) for (k,a) in self.activities.items()}        
        self._framerate = fps
        return self

    def tracks(self):
        """Return a dictionary of tracked object instances in the video scene"""        
        return self.tracks
    
    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame of video, save annotation visualization to provided outfile"""
        return self.__getitem__(frame).savefig(outfile if outfile is not None else temppng())
            
    def trim(self, startframe, endframe):
        """FIXME: the startframe and endframe should be set by the constructor if no arguments since this is set by the annotator"""
        super(Scene, self).trim(startframe, endframe)
        self.tracks = {k:t.offset(dt=-startframe) for (k,t) in self.tracks.items()}
        self.activities = {k:a.offset(dt=-startframe) for (k,a) in self.activities.items()}        
        return self

    def clip(self, startframe, endframe):
        """Alias for trim"""
        super(Scene, self).trim(startframe, endframe)
        self.tracks = {k:t.offset(dt=-startframe) for (k,t) in self.tracks.items()}
        self.activities = {k:a.offset(dt=-startframe) for (k,a) in self.activities.items()}                
        return self

    def crop(self, bb):
        """Crop the video using the supplied box, update tracks relative to crop, tracks may be outside the image rectangle"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        super(Scene, self).crop(bb)
        self.tracks = {k:t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for (k,t) in self.tracks.items()}
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self.tracks = {k:t.rot90ccw(H,W) for (k,t) in self.tracks.items()}
        super(Scene, self).rot90ccw()
        return self

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self.tracks = {k:t.rot90cw(H,W) for (k,t) in self.tracks.items()}
        super(Scene, self).rot90cw()
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self.tracks = {k:t.scalex(sx) for (k,t) in self.tracks.items()}
        self.tracks = {k:t.scaley(sy) for (k,t) in self.tracks.items()}
        super(Scene, self).resize(rows, cols)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def rescale(self, s):
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first then reload"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self.tracks = {k:t.rescale(s) for (k,t) in self.tracks.items()}
        super(Scene, self).rescale(s)
        return self

    def annotate(self, outfile=None):
        """Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, save as outfile.
        This function does not play the video, it only generates an annotation video.  Use show() to annotation and play."""
        assert self.isloaded(), "Call load() before annotate()"
        vid = self.load().clone()  # to save a new array
        outfile = outfile if outfile is not None else tempMP4()        
        (W, H) = (None, None)
        plt.close(1)
        for (k,im) in enumerate(self.__iter__()):
            imh = im.show(figure=1, nowindow=True)  # sets figure dimensions, does not display window
            if W is None or H is None:
                (W,H) = plt.figure(1).canvas.get_width_height()  # fast
                vid._array = np.zeros( (len(self), H, W,self.channels()), dtype=np.uint8)  # allocate
            buf = io.BytesIO()
            plt.figure(1).canvas.print_raw(buf)  # fast
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((H, W, 4))
            vid._array[k,:,:,:] = np.array(PIL.Image.fromarray(img).convert('RGB'))
        plt.close(1)
        return vid.saveas(outfile)

    def show(self, outfile=None, verbose=True):
        """Generate an annotation video saved to outfile (or tempfile if outfile=None) and show it using ffplay when it is done exporting"""
        outfile = tempMP4() if outfile is None else outfile
        if verbose:
            print('[vipy.video.show]: Generating annotation video "%s" ...' % outfile)
        self.annotate(outfile)
        cmd = "ffplay %s" % outfile
        if verbose:
            print('[vipy.video.show]: Executing "%s"' % cmd)
        os.system(cmd)
        return self
    
    
def RandomVideo(rows=None, cols=None, frames=None):
    """Return a random loaded vipy.video.video, useful for unit testing"""
    rows = np.random.randint(256, 1024) if rows is None else rows
    cols = np.random.randint(256, 1024) if cols is None else cols
    frames = np.random.randint(32, 256) if frames is None else frames
    assert rows>32 and cols>32 and frames>32    
    return Video(array=np.uint8(255 * np.random.rand(frames, rows, cols, 3)), colorspace='rgb')


def RandomScene(rows=None, cols=None, frames=None):
    """Return a random loaded vipy.video.Scene, useful for unit testing"""
    v = RandomVideo(rows, cols, frames)
    (rows, cols) = v.shape()
    tracks = [vipy.object.Track(label='track%d' % k, shortlabel='t%d' % k,
                                keyframes=[0, np.random.randint(50,100), np.random.randint(50,150)],
                                boxes=[vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,32)]

    activities = [vipy.object.Activity(label='activity%d' % k, shortlabel='a%d' % k, objectids=[tracks[np.random.randint(32)].id()], startframe=np.random.randint(50,100), endframe=np.random.randint(100,150)) for k in range(0,32)]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)

    return ims
    

def RandomSceneActivity(rows=None, cols=None, frames=256):
    """Return a random loaded vipy.video.Scene, useful for unit testing"""    
    v = RandomVideo(rows, cols, frames)
    (rows, cols) = v.shape()
    tracks = [vipy.object.Track(label=['Person','Vehicle','Object'][k], shortlabel='track%d' % k, boundary='strict', 
                                keyframes=[0, np.random.randint(50,100), np.random.randint(50,150)],
                                boxes=[vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,3)]

    activities = [vipy.object.Activity(label='Person Carrying', shortlabel='Carry', objectids=[tracks[0].id(), tracks[1].id()], startframe=np.random.randint(20,50), endframe=np.random.randint(70,100))]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)

    return ims
    

