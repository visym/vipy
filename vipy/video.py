import os
import dill
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng, \
    istuple, islist, isnumber, tolist, filefull, fileext, isS3url, totempdir, flatlist
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
import platform


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
    def __init__(self, filename=None, url=None, framerate=None, attributes=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None):
        self._url = None
        self._filename = None
        self._array = None
        self._colorspace = None
        self._ffmpeg = None
        self._framerate = None
        
        self.attributes = attributes if attributes is not None else {}
        assert filename is not None or url is not None or array is not None, 'Invalid constructor - Requires "filename", "url" or "array"'

        # Constructor clips
        assert (startframe is not None and endframe is not None) or (startframe is None and endframe is None), "Invalid input - (startframe,endframe) are both required"
        assert (startsec is not None and endsec is not None) or (startsec is None and endsec is None), "Invalid input - (startsec,endsec) are both required"        
        (self._startframe, self._endframe) = (startframe, endframe)
        (self._startsec, self._endsec) = (startsec, endsec)        

        # Input filenames
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
            self._url = url
        if filename is not None:
            #self.filename(filename)
            self._filename = filename
        else:
            if isS3url(self._url):
                self._filename = totempdir(self._url)  # Preserve S3 Object ID
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            elif isyoutubeurl(self._url):
                self._filename = os.path.join(tempdir(), '%s' % self._url.split('?')[1])
            if 'VIPY_CACHE' in os.environ and self._filename is not None:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._filename))

        # Video filter chain
        self._ffmpeg = ffmpeg.input(self.filename())  # restore, no other filters
        if self._startframe is not None and self._endframe is not None:
            self.clip(self._startframe, self._endframe)
        if self._startsec is not None and self._endsec is not None:
            self.cliptime(self._startsec, self._endsec)
            
        if framerate is not None:
            self.framerate(framerate)
            self._framerate = framerate

        # Array input
        if array is not None:
            self.array(array)
            self.colorspace(colorspace)

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, color=%s" % (self.height(), self.width(), len(self), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if not self.isloaded() and self._startframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if self._framerate is not None:
            strlist.append('fps=%s' % str(self._framerate))
        return str('<vipy.video: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames when the video is loaded, otherwise 0"""
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

    def _update_ffmpeg(self, argname, argval):
        nodes = ffmpeg.nodes.get_stream_spec_nodes(self._ffmpeg)
        sorted_nodes, outgoing_edge_maps = ffmpeg.dag.topo_sort(nodes)
        for n in sorted_nodes:
            if argname in n.__dict__['kwargs']:
                n.__dict__['kwargs'][argname] = argval
                return self
            else:
                print(n.__dict__)
        raise ValueError('invalid ffmpeg argument "%s" -> "%s"' % (argname, argval))
                
    def probe(self):
        """Run ffprobe on the filename and return the result as a JSON file"""
        assert self.hasfilename(), "Invalid video file '%s' for ffprobe" % self.filename() 
        return ffmpeg.probe(self.filename())
    
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
        assert not self.isloaded(), "Filters can only be applied prior to load()"
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

    def nourl(self):
        (self._url, self._urluser, self._urlpassword, self._urlsha1) = (None, None, None, None)
        return self

    def url(self, url=None, username=None, password=None, sha1=None):
        """Image URL and URL download properties"""
        if url is not None:
            self._url = url  # note that this does not change anything else, better to use the constructor for this
        if username is not None:
            self._urluser = username  # basic authentication
        if password is not None:
            self._urlpassword = password  # basic authentication
        if sha1 is not None:
            self._urlsha1 = sha1  # file integrity
        if url is None and username is None and password is None and sha1 is None:
            return self._url
        else:
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
    
    def reload(self):
        return self.clone(flush=True).load()
                       
    def nofilename(self):
        self._filename = None
        self._update_ffmpeg('filename', None)
        return self

    def filename(self, newfile=None):
        """Video Filename"""
        if newfile is None:
            return self._filename
        
        # Update ffmpeg filter chain with new input node filename
        newfile = os.path.abspath(os.path.expanduser(newfile))
        self._update_ffmpeg('filename', newfile)
        self._filename = newfile
        return self

    def download(self, ignoreErrors=False, timeout=10, verbose=True):
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
                vipy.videosearch.download(self._url, filefull(self._filename), writeurlfile=False, skip=ignoreErrors, verbose=verbose)
                for ext in ['mkv', 'mp4', 'webm']:
                    f = '%s.%s' % (self.filename(), ext)
                    if os.path.exists(f):
                        os.symlink(f, self.filename())  # for future load()
                        self.filename(f)
                        break    
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
                if self.filename() is None:
                    self.filename(totempdir(self._url))
                    if 'VIPY_CACHE' in os.environ:
                        self.filename(os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._url)))
                vipy.downloader.s3(self.url(), self.filename(), verbose=verbose)
                    
            elif url_scheme == 'scp':                
                if self.filename() is None:
                    self.filename(templike(self._url))                    
                    if 'VIPY_CACHE' in os.environ:
                        self.filename(os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._url)))
                vipy.downloader.scp(self._url, self.filename(), verbose=verbose)
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
        elif self.hasurl() and not self.hasfilename():
            self.download(verbose=True)  
        if not self.hasfilename():
            raise ValueError('Video file not found')
        im = Image(filename=tempjpg() if outfile is None else outfile)
        (out, err) = self._ffmpeg.output(im.filename(), vframes=1)\
                                 .overwrite_output()\
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True, capture_stderr=True)
        if not im.hasfilename():
            raise ValueError('Video preview failed - Attempted to load the video and no preview frame was loaded.  This usually occurs for zero length clips.') 
        return im

    def load(self, verbose=False, ignoreErrors=False, startframe=None, endframe=None, rotation=None, rescale=None, mindim=None):
        """Load a video using ffmpeg, applying the requested filter chain.  
           If verbose=True. then ffmpeg console output will be displayed. 
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
        elif not self.hasfilename() and self.hasurl():
            self.download(ignoreErrors=ignoreErrors)
        elif not self.hasfilename():
            raise ValueError('Invalid input - load() requires a valid URL, filename or array')
        if not self.hasfilename() and ignoreErrors:
            print('[vipy.video.load]: Video file "%s" not found - Ignoring' % self.filename())
            return self
        if True:
            print('[vipy.video.load]: Loading "%s"' % self.filename())

        # Increase filter chain from load() kwargs
        assert (startframe is not None and startframe is not None) or (startframe is None and endframe is None), "(startframe, endframe) must both be provided"
        if startframe is not None and endframe is not None:   
            self.clip(startframe, endframe)  # clip first
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
        imthumb = self._preview(verbose=verbose)
        (height, width, channels) = (imthumb.height(), imthumb.width(), imthumb.channels())
        (out, err) = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                                 .global_args('-loglevel', 'debug' if verbose else 'error') \
                                 .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, channels])  # read-only
        self.colorspace('rgb' if channels == 3 else 'lum')
        return self
    
    def clip(self, startframe, endframe):
        """Load a video clip betweeen start and end frames"""
        assert startframe <= endframe and startframe >= 0, "Invalid start and end frames (%s, %s)" % (str(startframe), str(endframe))
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe)\
                                   .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
        self._startframe = startframe if self._startframe is None else self._startframe + startframe  # for __repr__ only
        self._endframe = endframe if self._endframe is None else self._startframe + (endframe-startframe)  # for __repr__ only
        return self

    def cliptime(self, startsecs, endsecs):
        """Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly"""
        assert startsecs <= endsecs and startsecs >= 0, "Invalid start and end seconds (%s, %s)" % (str(startsecs), str(endsecs))
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.trim(start=startsecs, end=endsecs)\
                                   .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
        return self
    
    def rot90cw(self):
        """Rotate the video 90 degrees clockwise, can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('transpose', 1)
        return self

    def rot90ccw(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('transpose', 2)
        return self

    def rescale(self, s):
        """Rescale the video by factor s, such that the new dimensions are (s*H, s*W), can only be applied prior load()"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('scale', 'iw*%1.2f' % s, 'ih*%1.2f' % s)
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to be (rows, cols), can only be applied prior to load()"""
        if rows is None and cols is None:
            return self
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('scale', cols if cols is not None else -1, rows if rows is not None else -1)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def crop(self, bb):
        """Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        impreview = self._preview()  # to clip to image rectangle, required for ffmpeg
        bb = bb.imclipshape(impreview.width(), impreview.height())
        self._ffmpeg = self._ffmpeg.crop(bb.xmin(), bb.ymin(), bb.width(), bb.height())
        return self

    def saveas(self, outfile, framerate=30, vcodec='libx264', verbose=False, ignoreErrors=False):
        """Save video to new output video file.  This function does not draw boxes, it saves pixels to a new video file.

           * If self.array() is loaded, then export the contents of self._array to the video file
           * If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video
           * If outfile==None or outfile==self.filename(), then overwrite the current filename 
           * If ignoreErrors=True, then exit gracefully.  Useful for chaining download().saveas() on parallel dataset downloads

        """
        if True:
                print('[vipy.video.saveas]: Saving video "%s" ...' % outfile)                      
        try:
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
                # Requires saving to a tmpfile if the output filename is the same as the input filename
                tmpfile = '%s.tmp%s' % (filefull(outfile), fileext(outfile)) if outfile == self.filename() else outfile
                self._ffmpeg.filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                            .output(tmpfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                            .overwrite_output() \
                            .global_args('-loglevel', 'error' if not verbose else 'debug') \
                            .run()
                if outfile == self.filename():
                    if os.path.exists(self.filename()):
                        os.remove(self.filename())
                    shutil.move(tmpfile, self.filename())
            elif self.hasurl():
                raise ValueError('Input video url "%s" not downloaded, call download() first' % self.url())
            else:
                raise ValueError('Input video file not found "%s"' % self.filename())
        except Exception as e:
            if ignoreErrors:
                print('[vipy.video.saveas]:  Failed with error "%s" - Ignoring' % str(repr(e)))
            else:
                raise
            
        return outfile

    def save(self, ignoreErrors=False):
        """Save the current video filter chain, overwriting the current filename()"""
        return self.saveas(self.filename() if self.filename() is not None else tempMP4(), ignoreErrors=ignoreErrors)

    def pptx(self, outfile):
        """Export the video in a format that can be played by powerpoint"""
        pass

    def play(self, verbose=True):
        """Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it or try saveas(tempMP4())"""
        f = self.filename()
        if not self.isdownloaded():
            f = self.download().filename()            
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
        """Export loaded video to tempfile and play()"""
        return os.system('ffplay %s' % self.saveas(tempMP4()))
    
    def torch(self, take=None):
        """Convert the loaded video to an NxCxHxW torch tensor, forces a load()"""
        try_import('torch'); import torch

        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        self.load()
        frames = self._array if self.iscolor() else np.expand_dims(self._array, 3)
        t = torch.from_numpy(frames.transpose(0,3,1,2))
        return t if take is None else t[::int(np.round(len(t)/float(take)))][0:take]

    def clone(self, flushforward=False, flushbackward=False, flush=False):
        """Create deep copy of video object, flushing the original buffer if requested and returning the cloned object.
        Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned 
        object which can be used for encoding and will be garbage collected.
        
            * flushforward: copy the object, and set the cloned object array() to None.  This flushes the video buffer for the clone, not the object
            * flushbackward:  copy the object, and set the object array() to None.  This flushes the video buffer for the object, not the clone.
            * flush:  set the object array() to None and clone the object.  This flushes the video buffer for both the clone and the object.

        """
        if flush or (flushforward and flushbackward):
            self._array = None  # flushes buffer on object and clone
            im = copy.deepcopy(self)  # object and clone are flushed
        elif flushbackward:
            im = copy.deepcopy(self)  # propagates _array to clone
            self._array = None   # object flushed, clone not flushed
        elif flushforward:
            array = self._array;
            self._array = None
            im = copy.deepcopy(self)   # does not propagate _array to clone
            self._array = array    # object not flushed
            im._array = None   # clone flushed
        else:
            im = copy.deepcopy(self)            
        return im

    def flush(self):
        """Alias for clone(flush=True), returns self not clone"""
        self.clone(flush=True)
        return self

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

    A VideoCategory is a video with associated category, such as an activity class.  This class includes all of the constructors of vipy.video.Video 
    along with the ability to extract a clip based on frames or seconds.

    """
    def __init__(self, filename=None, url=None, framerate=30, attributes=None, category=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None):
        super(VideoCategory, self).__init__(url=url, filename=filename, framerate=framerate, attributes=attributes, array=array, colorspace=colorspace,
                                            startframe=startframe, endframe=endframe, startsec=startsec, endsec=endsec)
        self._category = category                
        
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
        if not self.isloaded() and self._startsec is not None:
            strlist.append('cliptime=(%1.2f,%1.2f)' % (self._startsec, self._endsec))
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
        
    def __init__(self, filename=None, url=None, framerate=None, array=None, colorspace=None, category=None, tracks=None, activities=None,
                 attributes=None, startframe=None, endframe=None, startsec=None, endsec=None):

        self._tracks = {}
        self._activities = {}        
        super(Scene, self).__init__(url=url, filename=filename, framerate=None, attributes=attributes, array=array, colorspace=colorspace,
                                    category=category, startframe=startframe, endframe=endframe, startsec=startsec, endsec=endsec)

        if tracks is not None:
            tracks = tracks if isinstance(tracks, list) or isinstance(tracks, tuple) else [tracks]  # canonicalize
            assert all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid track input; tracks=[vipy.object.Track(), ...]"
            self._tracks = {t.id():t for t in tracks}

        if activities is not None:
            activities = activities if isinstance(activities, list) or isinstance(activities, tuple) else [activities]  # canonicalize            
            assert all([isinstance(a, vipy.object.Activity) for a in activities]), "Invalid activity input; activities=[vipy.object.Activity(), ...]"
            self._activities = {a.id():a for a in activities}

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
        if not self.isloaded() and self._startframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if not self.isloaded() and self._startsec is not None:
            strlist.append('cliptime=(%1.2f,%1.2f)' % (self._startsec, self._endsec))            
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.hastracks():
            strlist.append('objects=%d' % len(self._tracks))
        if self.hasactivities():
            strlist.append('activities=%d' % len(self._activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        """Return the vipy.image.Scene() for the vipy.video.Scene() interpolated at frame k"""
        if self.load().isloaded() and k >= 0 and k < len(self):
            dets = [t[k] for (tid,t) in self._tracks.items() if t[k] is not None]  # track interpolation with boundary handling
            for d in dets:
                for (aid, a) in self._activities.items():
                    if a.hastrack(d.attributes['trackid']) and a.during(k):
                        # Shortlabel is displayed as "Noun Verb" during activity (e.g. Person Carry, Object Carry)
                        # Category is set to activity label during activity (e.g. all tracks in this activity have same color)
                        d.category(a.category())  # category label defines colors, see d.attributes['track'] for original labels 
                        d.shortlabel('%s %s' % (d.shortlabel(), a.shortlabel()))  # see d.attributes['track'] for original labels
                        if 'activity' not in d.attributes:
                            d.attributes['activity'] = []                            
                        d.attributes['activity'].append(a)  # for activity correspondence
            dets = sorted(dets, key=lambda d: d.shortlabel())   # layering in video is in alphabetical order of shortlabel
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

    def frame(self, k):
        """Alias for self[k]"""
        return self.__getitem__(k)
    
    def quicklook(self, n=9, dilate=1.5, mindim=256, fontsize=10):
        """Generate a montage of n uniformly spaced annotated frames centered on the union of the labeled boxes in the current frame to show the activity ocurring in this scene at a glance
           Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame
        """
        if not self.isloaded():
            self.mindim(mindim).load()
        framelist = [int(np.round(f)) for f in np.linspace(0, len(self)-1, n)]
        imframes = [self.frame(k).padcrop(self.frame(k).boundingbox().maxsquare().dilate(dilate)).mindim(mindim, interp='nearest') if (self.frame(k).boundingbox() is not None) else
                    self.frame(k).maxsquare() for k in framelist]
        imframes = [im.savefig(fontsize=fontsize).rgb() for im in imframes]
        return vipy.visualize.montage(imframes, imgwidth=mindim, imgheight=mindim)
        
    def tracks(self, tracks=None, id=None):
        if tracks is None:
            return self._tracks  # mutable
        elif id is not None:
            return self._tracks[id]
        else:
            assert all([isinstance(t, vipy.object.Track) for t in tolist(tracks)]), "Invalid input - Must be vipy.object.Track or list of vipy.object.Track"
            self._tracks = {t.id():t for t in tolist(tracks)}  # overwrite
            return self

    def activities(self, activities=None, id=None):
        if activities is None:
            return self._activities  # mutable
        elif id is not None:
            return self._activities[id]
        else:
            assert all([isinstance(a, vipy.object.Activity) for a in tolist(activities)]), "Invalid input - Must be vipy.object.Activity or list of vipy.object.Activities"
            self._activities = {a.id():a for a in tolist(activities)}   # overwrite
            return self

    def categories(self):
        """Return a set of all categories in all activities and tracks in this sccene"""
        return set([a.category() for a in self.activities().values()]+[t.category() for t in self.tracks().values()])
        
    def hasactivities(self):
        return len(self._activities) > 0

    def hastracks(self):
        return len(self._tracks) > 0

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
                scene.track(t2).add( ... )
        
        This will keep track of the current frame in the video and add the objects in the appropriate place

        """
        if isinstance(obj, vipy.object.Detection):
            assert self._currentframe is not None, "add() for vipy.object.Detection() must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=obj.category(), keyframes=[self._currentframe], boxes=[obj], boundary='strict', attributes=obj.attributes)
            self._tracks[t.id()] = t
            return t.id()
        elif isinstance(obj, vipy.object.Track):
            self._tracks[obj.id()] = obj
            return obj.id()
        elif isinstance(obj, vipy.object.Activity):
            self._activities[obj.id()] = obj
            return obj.id()
        elif (istuple(obj) or islist(obj)) and len(obj) == 4 and isnumber(obj[0]):
            assert self._currentframe is not None, "add() for obj=xywh must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=category, keyframes=[self._currentframe], boxes=[vipy.geometry.BoundingBox(xywh=obj)], boundary='strict', attributes=attributes)
            self._tracks[t.id()] = t
            return t.id()
        else:
            raise ValueError('Undefined object type "%s" to be added to scene - Supported types are obj in ["vipy.object.Detection", "vipy.object.Track", "vipy.object.Activity", "[xmin, ymin, width, height]"]' % str(type(obj)))        
        
    def dict(self):
        d = super(Scene, self).dict()
        d['category'] = self.category()
        d['tracks'] = [t.dict() for t in self._tracks.values()]
        d['activities'] = [a.dict() for a in self._activities.values()]
        return d
        
    def framerate(self, fps):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"        
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
        self._tracks = {k:t.framerate(fps) for (k,t) in self._tracks.items()}
        self._activities = {k:a.framerate(fps) for (k,a) in self._activities.items()}        
        self._framerate = fps
        return self

    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile"""
        return self.__getitem__(frame).savefig(outfile if outfile is not None else temppng())
        
    def activityclip(self, padframes=0):
        """Return a list of vipy.video.Scene() each clipped to be centered on a single activity, with an optional padframes before and after.  The Scene() category is updated to be the activity, and only the objects participating in the activity are included"""
        vid = self.clone(flushforward=True)
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity clips with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))
        activities = [a.clone() for a in vid.activities().values() if (a.endframe()-a.startframe()) > 0]   # only activities with at least one frame
        tracks = [ [t.clone() for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in activities]                         
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        padframes = padframes if istuple(padframes) else (padframes,padframes)
        return [vid.clone().activities(a).tracks(t).clip(startframe=max(a.startframe()-padframes[0], 0),
                                                         endframe=(a.endframe()+padframes[1])).category(a.category()) for (a,t) in zip(activities, tracks)]
    
    def activitycrop(self, dilate=1.0):
        """Returns a list of vipy.video.Scene() each spatially cropped to be the union of the objects performing the activity"""
        vid = self.clone(flushforward=True)
        activities = vid.activities().values()
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity clips with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))            
        tracks = [ [t for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in activities]                 
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        return [vid.clone().activities(a).tracks(t).crop(a.boundingbox().dilate(dilate)) for (a,t) in zip(activities, tracks)]        

    def activitysquare(self, dilate=1.0):
        """Returns a list of vipy.video.Scene() each spatially cropped to be the maxsquare of the union of the objects performing the activity"""
        vid = self.clone(flushforward=True)
        activities = vid.activities().values()
        tracks = [ [t for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in activities]                 
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        im = self._preview()  # for faster crop
        return [vid.clone().activities(a).tracks(t).crop(a.boundingbox().dilate(dilate).maxsquare().iminterior(im.width(), im.height())) for (a,t) in zip(activities, tracks)]  

    def activitytube(self, dilate=1.0, padframes=0):
        """Return a list of vipy.video.Scene() each spatially cropped following activitycrop() and temporally cropped following activityclip()"""
        return flatlist([a.activitycrop(dilate) for a in self.activityclip(padframes)])
        
    def clip(self, startframe, endframe):
        """Clip the video to between (startframe, endframe).  This clip is relative to cumulative clip() from the filter chain"""
        super(Scene, self).clip(startframe, endframe)
        self._tracks = {k:t.offset(dt=-startframe) for (k,t) in self._tracks.items()}
        self._activities = {k:a.offset(dt=-startframe) for (k,a) in self._activities.items()}        
        return self

    def cliptime(self, startsec, endsec):
        raise NotImplementedError('use clip() instead')
    
    def crop(self, bb):
        """Crop the video using the supplied box, update tracks relative to crop, bbox is clipped to be within the image rectabnle, otherwise ffmpeg will throw an exception"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        (H,W) = self._preview().shape()  # to clip to image rectangle        
        super(Scene, self).crop(bb.imclipshape(W,H))
        self._tracks = {k:t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for (k,t) in self._tracks.items()}
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90ccw(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).rot90ccw()
        return self

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90cw(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).rot90cw()
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self._tracks = {k:t.scalex(sx) for (k,t) in self._tracks.items()}
        self._tracks = {k:t.scaley(sy) for (k,t) in self._tracks.items()}
        super(Scene, self).resize(rows, cols)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def rescale(self, s):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self._preview().load().shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rescale(s) for (k,t) in self._tracks.items()}
        super(Scene, self).rescale(s)
        return self

    def annotate(self, outfile=None, n_processes=1, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[]):
        """Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, save as outfile and return a new video object where the frames contain the overlay.
        This function does not play the video, it only generates an annotation video.  Use show() which is equivalent to annotate().play()
        In general, this function should not be run on very long videos, as it requires loading the video framewise into memory, try running on clips instead.
        """
        outfile = outfile if outfile is not None else tempMP4()        
        if verbose:
            print('[vipy.video.annotate]: Generating annotation video "%s" ...' % outfile)
            if not self.isloaded():
                print('[vipy.video.annotate]: Loading video ...')  
        vid = self.load().clone()  # to save a new array
        assert self.isloaded(), "Load() failed"        
        if verbose:
                print('[vipy.video.annotate]: Annotating video ...')              
        if n_processes > 1:
            import vipy.batch
            with vipy.batch.Batch(vid, n_processes=n_processes) as b:
                print('[vipy.video.annotate.debug]: %s' % str(b))  # TESTING
                imgs = b.map(lambda v,k: v[k].savefig(fontsize=fontsize, 
                                                      captionoffset=captionoffset, 
                                                      textfacecolor=textfacecolor, 
                                                      textfacealpha=textfacealpha, 
                                                      shortlabel=shortlabel, 
                                                      boxalpha=boxalpha, 
                                                      d_category2color=d_category2color,
                                                      categories=categories, 
                                                      nocaption=nocaption,
                                                      nocaption_withstring=nocaption_withstring).rgb().numpy(), args=[(k,) for k in range(0, len(vid))])
            vid._array = np.stack(imgs, axis=0)            
        else:
            imgs = [vid[k].savefig(fontsize=fontsize,
                                   captionoffset=captionoffset,
                                   textfacecolor=textfacecolor,
                                   textfacealpha=textfacealpha,
                                   shortlabel=shortlabel,
                                   boxalpha=boxalpha,
                                   d_category2color=d_category2color,
                                   categories=categories,
                                   nocaption=nocaption,
                                   nocaption_withstring=nocaption_withstring).numpy() for k in range(0, len(vid))]  # SLOW for large videos
            vid._array = np.stack([np.array(PIL.Image.fromarray(img).convert('RGB')) for img in imgs], axis=0)
        return vid.filename(vid.saveas(outfile))


    def show(self, outfile=None, verbose=True, n_processes=1, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[]):
        """Generate an annotation video saved to outfile (or tempfile if outfile=None) and show it using ffplay when it is done exporting"""
        outfile = tempMP4() if outfile is None else outfile
        self.annotate(outfile, 
                      n_processes=n_processes, 
                      verbose=verbose, 
                      fontsize=fontsize,
                      captionoffset=captionoffset,
                      textfacecolor=textfacecolor,
                      textfacealpha=textfacealpha,
                      shortlabel=shortlabel,
                      boxalpha=boxalpha,
                      d_category2color=d_category2color,
                      categories=categories,
                      nocaption=nocaption, 
                      nocaption_withstring=nocaption_withstring)
        cmd = "ffplay %s" % outfile
        if verbose:
            print('[vipy.video.show]: Executing "%s"' % cmd)
        os.system(cmd)
        return self
    
    
def RandomVideo(rows=None, cols=None, frames=None):
    """Return a random loaded vipy.video.video, useful for unit testing, minimum size (32x32x32)"""
    rows = np.random.randint(256, 1024) if rows is None else rows
    cols = np.random.randint(256, 1024) if cols is None else cols
    frames = np.random.randint(32, 256) if frames is None else frames
    assert rows>32 and cols>32 and frames>=32    
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

    activities = [vipy.object.Activity(label='activity%d' % k, shortlabel='a%d' % k, tracks={tracks[j].id():tracks[j] for j in [np.random.randint(32)]}, startframe=np.random.randint(50,100), endframe=np.random.randint(100,150)) for k in range(0,32)]   
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

    activities = [vipy.object.Activity(label='Person Carrying', shortlabel='Carry', tracks={tracks[0].id():tracks[0], tracks[1].id():tracks[1]}, startframe=np.random.randint(20,50), endframe=np.random.randint(70,100))]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)

    return ims
    

