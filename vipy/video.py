import os
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng
from vipy.image import Image
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
            else:
                self._filename = tempMP4()  # guess MP4 for URLs with no file extension
            if 'VIPY_CACHE' in os.environ:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._filename))
        if array is not None:
            self.array(array)
            self.colorspace(colorspace)
            
        # Transformations
        self._ffmpeg = ffmpeg.input(self.filename())        
        if framerate is not None:
            self.framerate(framerate)
            self._framerate = framerate
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, colorspace=%s" % (self.height(), self.width(), len(self), self.colorspace()))
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
            return Image(array=self._array[k], colorspace='rgb')
        elif not self.isloaded():
            raise ValueError('Video not loaded, load() before indexing')
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        """Iterate over frames, yielding vipy.image.Image object for each frame"""
        # FIXME: Streaming video access for large videos that will not fit into memory
        # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
        # FIXME: https://github.com/kkroening/ffmpeg-python/issues/78
        self.load()
        for k in range(0, len(self)):
            yield self.__getitem__(k)

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

    def array(self, array=None):
        if array is None:
            return self._array
        elif isnumpy(array):
            assert array.dtype == np.float32 or array.dtype == np.uint8, "Invalid input - array() must be type uint8 or float32"
            assert array.ndim == 4, "Invalid input array() must be of shape NxHxWxC, for N frames, of size HxW with C channels"
            self._array = np.copy(array)
            self._filename = None
            self._url = None
            self.colorspace(None)  # must be set with colorspace() before conversion
        else:
            raise ValueError('Invalid input - array() must be numpy array')            

    def tonumpy(self):
        return self._array

    def numpy(self):
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
            return self

    def download(self, ignoreErrors=False, timeout=10, verbose=False):
        """Download URL to filename provided by constructor, or to temp filename"""
        if self._url is None and self._filename is not None:
            return self
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[vipy.video.download][ERROR]: Invalid URL "%s" ' % self._url)

        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if isyoutubeurl(self._url):
                vipy.videosearch.download(self._url, self._filename, writeurlfile=False, skip=ignoreErrors)
                if not self.hasfilename():
                    for ext in ['mkv', 'mp4', 'webm']:
                        f = '%s.%s' % (self.filename(), ext)
                        if os.path.exists(f):
                            os.symlink(f, self.filename())  # for ffmpeg-python filters, yuck
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
        im = Image(filename=tempjpg() if outfile is None else outfile)
        (out, err) = self._ffmpeg.output(im.filename(), vframes=1)\
                                 .overwrite_output()\
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True, capture_stderr=True)
        return im

    def load(self, verbosity=1, ignoreErrors=False):
        """Load a video using ffmpeg, applying the requested filter chain.  If verbosity=2. then ffmpeg console output will be displayed. If ignoreErrors=True, then download errors are warned and skipped"""
        if self.isloaded():
            return self
        elif not self.hasfilename() and not self.isloaded():
            self.download(ignoreErrors=ignoreErrors)
        if not self.hasfilename() and ignoreErrors:
            print('[vipy.video.load]: Video file "%s" not found - Ignoring' % self.filename())
            return self
        if verbosity > 0:
            print('[vipy.video.load]: Loading "%s"' % self.filename())

        # Generate single frame _preview to get frame sizes
        imthumb = self._preview(verbose=verbosity > 1)
        (height, width, channels) = (imthumb.height(), imthumb.width(), imthumb.channels())
        (out, err) = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                                 .global_args('-loglevel', 'debug' if verbosity > 1 else 'error') \
                                 .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, channels])
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

    def crop(self, bb):
        """Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load()"""
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        self._ffmpeg = self._ffmpeg.crop(bb.xmin(), bb.ymin(), bb.width(), bb.height())
        return self

    def saveas(self, outfile, framerate=30, vcodec='libx264', verbose=False):
        """Save video to new output video file, from either numpy buffer in self._array after calling load(), or by applying filter chain if not loaded"""
        if self.isloaded():
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
        elif self.isdownloaded():
            self._ffmpeg.output(outfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
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

    def play(self):
        """Play the saved video filename using the system 'ffplay'"""
        if not self.isdownloaded():
            self.download()
        cmd = "ffplay %s" % self.filename()
        print('[vipy.video.play]: %s' % cmd)
        os.system(cmd)
        return self

    def torch(self, take=None):
        """Convert the loaded video to an NxCxHxW torch tensor"""
        try_import('torch'); import torch

        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        frames = self._array if self.iscolor() else np.expand_dims(self._array, 3)
        t = torch.from_numpy(frames.transpose(0,3,1,2))
        return t if take is None else t[::int(np.round(len(t)/float(take)))][0:take]

    def clone(self):
        """Copy the video object"""
        return copy.deepcopy(self)


class Scene(Video):
    """ vipy.video.Scene class
    """
        
    def __init__(self, filename=None, url=None, framerate=None, attributes=None, tracks=None, activities=None):
        super(Scene, self).__init__(url=url, filename=filename, framerate=None, attributes=attributes)

        self._tracks = []
        if tracks is not None:
            assert isinstance(tracks, list) and all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid input"
            self._tracks = tracks

        self._activities = []
        if activities is not None:
            assert isinstance(activities, list) and all([isinstance(a, vipy.activity.Activity) for a in activities]), "Invalid input"
            self._activities = activities

        if framerate is not None:
            self.framerate(framerate)
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self._framerate is not None:
            strlist.append('fps=%s' % str(self._framerate))            
        if len(self._tracks) > 0:
            strlist.append('tracks=%d' % len(self._tracks))
        if len(self._activities) > 0:
            strlist.append('activities=%d' % len(self._activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        self.load()
        if k >= 0 and k < len(self):
            return vipy.image.Scene(array=self._array[k], colorspace='rgb', objects=[t[k] for t in self._tracks])
        elif not self.isloaded():
            raise ValueError('Video not loaded, load() before indexing')
        else:
            raise ValueError('Invalid frame index %d ' % k)

    def __iter__(self):
        self.load()
        for k in range(0, len(self)):
            yield self.__getitem__(k)

    def framerate(self, fps):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        assert not self.isloaded(), "Filters can only be applied prior to loading; flush() the video first"        
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
        self._tracks = [t.framerate(fps) for t in self._tracks]
        self._framerate = fps
        return self

    def objects(self):
        """Return a list of objects in the video scene"""
        return self._tracks

    def tracks(self):
        """Alias for objects"""
        return self._tracks
    
    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame of video, save annotation visualization to provided outfile"""
        return self.__getitem__(frame).savefig(outfile if outfile is not None else temppng())
            
    def trim(self, startframe, endframe):
        """FIXME: the startframe and endframe should be set by the constructor if no arguments since this is set by the annotator"""
        super(Scene, self).trim(startframe, endframe)
        self._tracks = [t.offset(dt=-startframe) for t in self._tracks]
        return self

    def clip(self, startframe, endframe):
        """Alias for trim"""
        super(Scene, self).trim(startframe, endframe)
        self._tracks = [t.offset(dt=-startframe) for t in self._tracks]
        return self

    def crop(self, bb):
        """Crop the video using the supplied box, update tracks relative to crop, tracks may be outside the image rectangle"""
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        super(Scene, self).crop(bb)
        self._tracks = [t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for t in self._tracks]
        return self

    def rot90ccw(self):
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = [t.rot90ccw(H,W) for t in self._tracks]
        super(Scene, self).rot90ccw()
        return self

    def rot90cw(self):
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = [t.rot90cw(H,W) for t in self._tracks]
        super(Scene, self).rot90cw()
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided"""
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self._tracks = [t.scalex(sx) for t in self._tracks]
        self._tracks = [t.scaley(sy) for t in self._tracks]
        super(Scene, self).resize(rows, cols)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def rescale(self, s):
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
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
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((H, W, 4))
            vid._array.append(np.array(PIL.Image.fromarray(img).convert('RGB')))
        plt.close(1)

        vid._array = np.array(vid._array)
        return vid.saveas(outfile)


class VideoCategory(Video):
    """vipy.video.VideoCategory class
    """
    def __init__(self, filename=None, url=None, framerate=30, attributes=None, category=None, startframe=None, endframe=None):
        super(VideoCategory, self).__init__(url=url, filename=filename, framerate=framerate, attributes=attributes)
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
        if self._category is not None:
            strlist.append('category="%s"' % self.category())
        if not self.isloaded() and self._startframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        return str('<vipy.video.VideoCategory: %s>' % (', '.join(strlist)))

    def category(self, c=None):
        if c is None:
            return self._category
        else:
            self._category = c
            return self
