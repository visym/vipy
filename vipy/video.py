import os
import dill
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng, \
    istuple, islist, isnumber, tolist, filefull, fileext, isS3url, totempdir, flatlist, tocache, premkdir, writecsv
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
import uuid
import platform
from io import BytesIO
import vipy.globals
import vipy.activity


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
    def __init__(self, filename=None, url=None, framerate=30.0, attributes=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None):
        self._url = None
        self._filename = None
        self._array = None
        self._colorspace = None
        self._ffmpeg = None
        self._framerate = framerate
        
        self.attributes = attributes if attributes is not None else {}
        assert filename is not None or url is not None or array is not None, 'Invalid constructor - Requires "filename", "url" or "array"'

        # FFMPEG installed?
        ffmpeg_exe = shutil.which('ffmpeg')
        ffprobe_exe = shutil.which('ffprobe')        
        ffplay_exe = shutil.which('ffplay')        
        if ffmpeg_exe is None or not os.path.exists(ffmpeg_exe):
            warnings.warn('"ffmpeg" executable not found on path, this is required for vipy.video - Install from http://ffmpeg.org/download.html')
        if ffprobe_exe is None or not os.path.exists(ffprobe_exe):
            warnings.warn('"ffprobe" executable not found on path, this is optional for vipy.video - Install from http://ffmpeg.org/download.html')            
        if ffplay_exe is None or not os.path.exists(ffplay_exe):
            warnings.warn('"ffplay" executable not found on path, this is used for visualization and is optional for vipy.video - Install from http://ffmpeg.org/download.html')            

        # Constructor clips
        assert (startframe is not None and endframe is not None) or (startframe is None and endframe is None), "Invalid input - (startframe,endframe) are both required"
        assert (startsec is not None and endsec is not None) or (startsec is None and endsec is None), "Invalid input - (startsec,endsec) are both required"        
        (self._startframe, self._endframe) = (None, None)  # __repr__ only
        (self._startsec, self._endsec) = (None, None)      # __repr__ only  

        # Input filenames
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
            self._url = url
        if filename is not None:
            self._filename = filename
        elif self._url is not None:
            if isS3url(self._url):
                self._filename = totempdir(self._url)  # Preserve S3 Object ID
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            elif isyoutubeurl(self._url):
                self._filename = os.path.join(tempdir(), '%s' % self._url.split('?')[1])
            else:
                self._filename = totempdir(self._url)  
            if 'VIPY_CACHE' in os.environ and self._filename is not None:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._filename))

        # Video filter chain
        self._ffmpeg = ffmpeg.input(self.filename())  # restore, no other filters
        if framerate is not None:
            self.framerate(framerate)
            self._framerate = framerate        
        if startframe is not None and endframe is not None:
            self.clip(startframe, endframe)  
        if startsec is not None and endsec is not None:
            (self._startsec, self._endsec) = (startsec, endsec)            
            self.cliptime(startsec, endsec)
            
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
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if self._framerate is not None:
            strlist.append('fps=%1.1f' % float(self._framerate))
        return str('<vipy.video: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames in the video if loaded, else zero.  Do not automatically trigger a load, since this can interact in unexpected ways with other tools that depend on fast __len__()"""
        if not self.isloaded():
            warnings.warn('Load() video to see number of frames - Returning zero')  # should this just throw an exception?
        return len(self.array()) if self.isloaded() else 0

    def __getitem__(self, k):
        """Return the kth frame as an vipy.image object"""
        assert isinstance(k, int), "Indexing video by frame must be integer"        
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
        raise ValueError('invalid ffmpeg argument "%s" -> "%s"' % (argname, argval))
               
    def _ffmpeg_commandline(self, f=None):
        """Return the ffmpeg command line string that will be used to process the video"""
        cmd = f.compile() if f is not None else self._ffmpeg.output('vipy_output.mp4').compile()
        for (k,c) in enumerate(cmd):
            if c is None:
                cmd[k] = str(c)
            elif 'filter' in c:
                cmd[k+1] = '"%s"' % str(cmd[k+1])
            elif 'map' in c:
                cmd[k+1] = '"%s"' % str(cmd[k+1])
        return str(' ').join(cmd)

    def probe(self):
        """Run ffprobe on the filename and return the result as a JSON file"""
        assert self.hasfilename(), "Invalid video file '%s' for ffprobe" % self.filename() 
        return ffmpeg.probe(self.filename())

    def print(self, prefix='', verbose=True):
        """Print the representation of the video - useful for debugging in long fluent chains"""
        if verbose:
            print(prefix+self.__repr__())
        return self

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
                 'ffmpeg':self._ffmpeg_commandline(),
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

    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        if fps is None:
            return self._framerate
        else:
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
        if not self.isloaded():
            previewhash = hash(str(self._ffmpeg.output('dummyfile').compile()))
            if not hasattr(self, '_previewhash') or previewhash != self._previewhash:
                im = self._preview()  # ffmpeg chain changed, load a single frame of video 
                self._channels = im.channels()  # cache
                self._previewhash = previewhash
            return self._channels  # cached
        else:
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
    
    def zeros(self):
        self._array = 0*self.load()._array
        return self

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
        newfile = os.path.normpath(os.path.abspath(os.path.expanduser(newfile)))
        self._update_ffmpeg('filename', newfile)
        self._filename = newfile
        return self

    def filesize(self):
        """Return the size in bytes of the filename(), None if the filename() is invalid"""
        return os.path.getsize(self.filename()) if self.hasfilename() else None

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
            
            elif url_scheme in ['http', 'https'] and isvideourl(self._url):
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
 
            elif not isvideourl(self._url) and vipy.videosearch.is_downloadable_url(self._url):
                vipy.videosearch.download(self._url, filefull(self._filename), writeurlfile=False, skip=ignoreErrors, verbose=verbose)
                for ext in ['mkv', 'mp4', 'webm']:
                    f = '%s.%s' % (self.filename(), ext)
                    if os.path.exists(f):
                        os.symlink(f, self.filename())  # for future load()
                        self.filename(f)
                        break    
                if not self.hasfilename():
                    raise ValueError('Downloaded filenot found "%s.*"' % self.filename())
                
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

    def fetch(self, ignoreErrors=False):
        """Download only if hasfilename() is not found"""
        return self.download(ignoreErrors=ignoreErrors) if not self.hasfilename() else self

    def shape(self):
        """Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded"""
        if not self.isloaded():
            previewhash = hash(str(self._ffmpeg.output('dummyfile').compile()))
            if not hasattr(self, '_previewhash') or previewhash != self._previewhash:
                im = self._preview()  # ffmpeg chain changed, load a single frame of video 
                self._shape = (im.height(), im.width())  # cache the shape
                self._previewhash = previewhash
            return self._shape
        else:
            return (self._array.shape[1], self._array.shape[2])

    def width(self):
        """Width (cols) in pixels of the video for the current filter chain"""
        return self.shape()[1]

    def height(self):
        """Height (rows) in pixels of the video for the current filter chain"""
        return self.shape()[0]

    def _preview(self, framenum=0):
        """Return selected frame of filtered video, return vipy.image.Image object.  This is useful for previewing the frame shape of a complex filter chain without loading the whole video."""
        if self.isloaded():
            return self[0]
        elif self.hasurl() and not self.hasfilename():
            self.download(verbose=True)  
        if not self.hasfilename():
            raise ValueError('Video file not found')

        # Convert frame to mjpeg and pipe to stdout
        try:
            f = self._ffmpeg.filter('select', 'gte(n,{})'.format(framenum))\
                            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')\
                            .global_args('-cpuflags', '0', '-loglevel', 'debug' if vipy.globals.verbose() else 'error')
            (out, err) = f.run(capture_stdout=True)            
        except Exception as e:
            raise ValueError('[vipy.video.load]: Video preview failed for video "%s" with ffmpeg command "%s" - Try manually running ffmpeg to see errors' % (str(self), str(self._ffmpeg_commandline(f))))

        # [EXCEPTION]:  UnidentifiedImageError: cannot identify image file
        #   -This may occur when the framerate of the video from ffprobe (tbr) does not match that passed to fps filter, resulting in a zero length image preview piped to stdout
        return Image(array=np.array(PIL.Image.open(BytesIO(out))))

    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile"""
        return self.__getitem__(frame).savefig(outfile if outfile is not None else temppng())
    
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
        if verbose:
            print('[vipy.video.load]: Loading "%s"' % self.filename())

        # Increase filter chain from load() kwargs
        assert (startframe is not None and startframe is not None) or (startframe is None and endframe is None), "(startframe, endframe) must both be provided"
        if startframe is not None and endframe is not None:   
            self = self.clip(startframe, endframe)  # clip first
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
        imthumb = self._preview()
        (height, width, channels) = (imthumb.height(), imthumb.width(), imthumb.channels())

        # Load the video
        # 
        # [EXCEPTION]:  older ffmpeg versions may segfault on complex crop filter chains
        #    -On some versions of ffmpeg setting -cpuflags=0 fixes it, but the right solution is to rebuild from the head (30APR20)
        #
        try:
            f = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24')\
                            .global_args('-cpuflags', '0', '-loglevel', 'debug' if vipy.globals.verbose() else 'error')
            (out, err) = f.run(capture_stdout=True)
        except Exception as e:
            raise ValueError('[vipy.video.load]: Load failed for video "%s" with ffmpeg command "%s" - Try load(verbose=True) or manually running ffmpeg to see errors' % (str(self), str(self._ffmpeg_commandline(f))))

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

    def cliptime(self, startsec, endsec):
        """Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly"""
        assert startsec <= endsec and startsec >= 0, "Invalid start and end seconds (%s, %s)" % (str(startsec), str(endsec))
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.trim(start=startsec, end=endsec)\
                                   .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
        self._startsec = startsec if self._startsec is None else self._startsec + startsec  # for __repr__ only
        self._endsec = endsec if self._endsec is None else self._startsec + (endsec-startsec)  # for __repr__ only
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

    def fliplr(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('hflip')
        return self

    def flipud(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('vflip')
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
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def randomcrop(self, shape, withbox=False):
        """Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box"""
        assert shape[0] <= self.height() and shape[1] <= self.width()  # triggers preview()
        (xmin, ymin) = (np.random.randint(self.height()-shape[0]), np.random.randint(self.width()-shape[1]))
        bb = vipy.geometry.BoundingBox(xmin=xmin, ymin=ymin, width=shape[1], height=shape[0])  # may be outside frame
        self.crop(bb, zeropad=True)
        return self if not withbox else (self, bb)

    def centercrop(self, shape, withbox=False):
        """Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box"""
        assert shape[0] <= self.height() and shape[1] <= self.width()  # triggers preview()
        bb = vipy.geometry.BoundingBox(xcentroid=self.width()/2.0, ycentroid=self.height()/2.0, width=shape[1], height=shape[0]).int()  # may be outside frame
        self.crop(bb, zeropad=True)  
        return self if not withbox else (self, bb)

    def centersquare(self):
        """Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant"""
        return self.centercrop( (min(self.height(), self.width()), min(self.height(), self.width())))

    def maxsquare(self):
        # This ffmpeg filter can throw the error:  "Padded dimensions cannot be smaller than input dimensions." since the preview is off by one.  Add one here to make sure.
        # FIXME: not sure where in some filter chains this off-by-one error is being introduced, but probably does not matter since it does not affect any annotations 
        # since the max square always preserves the scale and the upper left corner of the source video. 
        self._ffmpeg = self._ffmpeg.filter('pad', max(self.shape())+1, max(self.shape())+1, 0, 0)  
        return self
        
    def zeropad(self, padwidth, padheight):
        """Zero pad the video with padwidth columns before and after, and padheight rows before and after"""
        assert isinstance(padwidth, int) and isinstance(padheight, int)
        self._ffmpeg = self._ffmpeg.filter('pad', 'iw+%d' % (2*padwidth), 'ih+%d' % (2*padheight), '%d'%padwidth, '%d'%padheight)
        return self

    def crop(self, bb, zeropad=True):
        """Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().
           If strict=False, then we do not perform bounds checking on this bounding box
        """
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        assert not bb.isdegenerate() and bb.isnonnegative() 
        bb = bb.int()
        if zeropad and bb != bb.clone().imclipshape(self.width(), self.height()):
            # Crop outside the image rectangle will segfault ffmpeg, pad video first (if zeropad=False, then rangecheck will not occur!)
            self.zeropad(bb.width(), bb.height())     # cannot be called in derived classes
            bb = bb.offset(bb.width(), bb.height())   # Shift boundingbox by padding
        self._ffmpeg = self._ffmpeg.filter('crop', '%d' % bb.width(), '%d' % bb.height(), '%d' % bb.xmin(), '%d' % bb.ymin(), 0, 1)  # keep_aspect=False, exact=True
        return self

    def saveas(self, outfile=None, framerate=None, vcodec='libx264', verbose=False, ignoreErrors=False, flush=False):
        """Save video to new output video file.  This function does not draw boxes, it saves pixels to a new video file.

           * If self.array() is loaded, then export the contents of self._array to the video file
           * If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video
           * If outfile==None or outfile==self.filename(), then overwrite the current filename 
           * If ignoreErrors=True, then exit gracefully.  Useful for chaining download().saveas() on parallel dataset downloads
           * Returns a new video object with this video filename, and a clean video filter chain
           * if flush=True, then flush this buffer right after saving the new video. This is useful for transcoding in parallel
           * framerate:  input framerate of the frames in the buffer, or the output framerate of the transcoded video.  If not provided, use framerate of source video
        """        
        outfile = tocache(tempMP4()) if outfile is None else outfile
        premkdir(outfile)  # create output directory for this file if not exists
        framerate = framerate if framerate is not None else self._framerate

        if verbose:
            print('[vipy.video.saveas]: Saving video "%s" ...' % outfile)                      
        try:
            if self.isloaded():
                # Save numpy() from load() to video, forcing to be even shape
                (n, height, width, channels) = self._array.shape
                process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate) \
                                .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                                .output(filename=outfile, pix_fmt='yuv420p', vcodec=vcodec) \
                                .overwrite_output() \
                                .global_args('-cpuflags', '0', '-loglevel', 'error' if not vipy.globals.verbose() else 'debug') \
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
                            .output(filename=tmpfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                            .overwrite_output() \
                            .global_args('-cpuflags', '0', '-loglevel', 'error' if not vipy.globals.verbose() else 'debug') \
                            .run()
                if outfile == self.filename():
                    if os.path.exists(self.filename()):
                        os.remove(self.filename())
                    shutil.move(tmpfile, self.filename())
            elif self.hasurl():
                raise ValueError('Input video url "%s" not downloaded, call download() first' % self.url())
            elif not self.isloaded():
                raise ValueError('Input video not loaded - Try calling load() first')
            elif not self.hasfilename():
                raise ValueError('Input video file not found "%s"' % self.filename())
            else: 
                raise ValueError('saveas() failed')
        except Exception as e:
            if ignoreErrors:
                # useful for saving a large number of videos in parallel where some failed download
                print('[vipy.video.saveas]:  Failed with error "%s" - Returning empty video' % str(repr(e)))
            else:
                raise

        # Return a new video, cloned from this video with the new video file, optionally flush the video we loaded before returning
        return self.clone(flushforward=True, flushfilter=True, flushbackward=flush).filename(outfile)
    
    def savetmp(self):
        return self.saveas(outfile=tempMP4())

    def pptx(self, outfile):
        """Export the video in a format that can be played by powerpoint"""
        pass

    def play(self, verbose=True):
        """Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it """
        v = self
        if not self.isdownloaded() and self.hasurl():
            v = self.download()
        if not self.hasfilename():
            v = self.saveas()  # save to temporary video         
        assert v.hasfilename(), "Video frames must be saved to file prior to play() - Try calling saveas() first"
        cmd = "ffplay %s" % v.filename()
        if verbose:
            print('[vipy.video.play]: Executing "%s"' % cmd)
        os.system(cmd)
        return self

    def show(self):
        """Alias for play()"""
        return self.play()
    
    def torch(self, startframe=0, endframe=None, length=None, stride=1, take=None, boundary='repeat', order='nchw', verbose=False, withslice=False, scale=1.0):
        """Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().
           Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.
           Follows numpy slicing rules.  Optionally return the slice used if withslice=True
           Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()           
        """
        try_import('torch'); import torch
        frames = self.load().array() if self.iscolor() else np.expand_dims(self.load().array(), 3)
        assert boundary in ['repeat', 'strict'], "Invalid boundary mode - must be in ['repeat', 'strict']"

        # Slice index (i=start, j=end, k=step)
        (i,j,k) = (startframe, len(frames), stride)
        if startframe == 'random':
            assert length is not None, "Random start frame requires fixed length"
            i = max(0, np.random.randint(len(frames)-length+1))
        if endframe is not None:
            assert length is None, "Cannot specify both endframe and length"                        
            assert endframe > startframe, "End frame must be greater than start frame"
            (j,k) = (endframe-startframe+1, 1)
        if length is not None:
            assert endframe is None, "Cannot specify both endframe and length"
            assert length >= 0, "Length must be positive"
            (j,k) = (i+length, 1)
        if stride != 1:
            assert take is None, "Cannot specify both take and stride"
            assert stride >= 1, "Stride must be >= 1"
            k = stride
        if take is not None:
            # Uniformly sampled frames to result in len(frames)=take
            assert stride == 1, "Cannot specify both take and stride"
            assert take <= len(frames), "Take must be less than the number of frames"
            k = int(np.ceil(len(frames)/float(take)))

        # Boundary handling
        assert i >= 0, "Start frame must be >= 0"
        assert i < j, "Start frame must be less then end frame"
        assert k <= len(frames), "Stride must be <= len(frames)"            
        if boundary == 'repeat' and j > len(frames):
            for d in range(j-len(frames)):
                frames = np.concatenate( (frames, np.expand_dims(frames[-1], 0) ))
        assert j <= len(frames), "invalid slice=%s for frame shape=%s - try setting boundary='repeat'" % (str((i,j,k)), str(frames.shape))
        if verbose:
            print('[vipy.video.torch]: slice (start,end,step)=%s for frame shape (N,C,H,W)=%s' % (str((i,j,k)), str(frames.shape)))

        # Slice and transpose to torch tensor axis ordering
        t = torch.from_numpy(frames[i:j:k])
        if order == 'nchw':
            t = t.permute(0,3,1,2)  # NxCxHxW
        elif order == 'nhwc':
            pass  # NxHxWxC  (native numpy order)
        else:
            raise ValueError("Invalid order = must be in ['nchw', 'nhwc']")
            
        # Scaling (optional)
        if scale is not None and self.colorspace() != 'float':
            t = (1.0/255.0)*t  # [0,255] -> [0,1]
        elif scale is not None and scale != 1.0:
            t = scale*t

        # Return tensor or (tensor, slice)
        return t if not withslice else (t, (i,j,k))

    def clone(self, flushforward=False, flushbackward=False, flush=False, flushfilter=False, rekey=False):
        """Create deep copy of video object, flushing the original buffer if requested and returning the cloned object.
        Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned 
        object which can be used for encoding and will be garbage collected.
        
            * flushforward: copy the object, and set the cloned object array() to None.  This flushes the video buffer for the clone, not the object
            * flushbackward:  copy the object, and set the object array() to None.  This flushes the video buffer for the object, not the clone.
            * flush:  set the object array() to None and clone the object.  This flushes the video buffer for both the clone and the object.
            * flushfilter:  Set the ffmpeg filter chain to the default in the new object, useful for saving new videos
            * rekey: Generate new unique track ID and activity ID keys for this scene
 
        """
        if flush or (flushforward and flushbackward):
            self._array = None  # flushes buffer on object and clone
            self._previewhash = None
            v = copy.deepcopy(self)  # object and clone are flushed
        elif flushbackward:
            v = copy.deepcopy(self)  # propagates _array to clone
            self._array = None   # object flushed, clone not flushed
            self._previewhash = None
        elif flushforward:
            array = self._array;
            self._array = None
            self._previewhash = None
            v = copy.deepcopy(self)   # does not propagate _array to clone
            self._array = array    # object not flushed
            v._array = None   # clone flushed
        else:
            v = copy.deepcopy(self)            
        if flushfilter:
            v._ffmpeg = ffmpeg.input(v.filename())  # no other filters
            v._previewhash = None
            (v._startframe, v._endframe) = (None, None)
            (v._startsec, v._endsec) = (None, None)
        if rekey:
            v.rekey()
        return v

    def flush(self):
        """Alias for clone(flush=True), returns self not clone"""
        self._array = None  # flushes buffer on object and clone
        self._previewhash = None
        return self

    def flush_and_return(self, retval):
        """Flush the video and return the parameter supplied, useful for long fluent chains"""
        self.flush()
        return retval

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

    def normalize(self, mean, std, scale=1.0):
        """Pixelwise whitening, out = ((scale*in) - mean) / std); triggers load()"""
        self._array = (((scale*self.load()._array) - np.array(mean)) / np.array(std)).astype(np.float32)
        self.colorspace('float')
        return self

    
class VideoCategory(Video):
    """vipy.video.VideoCategory class

    A VideoCategory is a video with associated category, such as an activity class.  This class includes all of the constructors of vipy.video.Video 
    along with the ability to extract a clip based on frames or seconds.

    """
    def __init__(self, filename=None, url=None, framerate=30.0, attributes=None, category=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None):
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
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
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
    activity performed by the object instances.  Track and activity timing must be relative to the start frame of the Scene() constructor.  

    """
        
    def __init__(self, filename=None, url=None, framerate=30.0, array=None, colorspace=None, category=None, tracks=None, activities=None,
                 attributes=None, startframe=None, endframe=None, startsec=None, endsec=None):

        self._tracks = {}
        self._activities = {}        
        super(Scene, self).__init__(url=url, filename=filename, framerate=framerate, attributes=attributes, array=array, colorspace=colorspace,
                                    category=category, startframe=startframe, endframe=endframe, startsec=startsec, endsec=endsec)

        # Tracks must be defined relative to the clip specified by this constructor
        if tracks is not None:
            tracks = tracks if isinstance(tracks, list) or isinstance(tracks, tuple) else [tracks]  # canonicalize
            assert all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid track input; tracks=[vipy.object.Track(), ...]"
            self._tracks = {t.id():t for t in tracks}

        # Activites must be defined relative to the clip specified by this constructor            
        if activities is not None:
            activities = activities if isinstance(activities, list) or isinstance(activities, tuple) else [activities]  # canonicalize            
            assert all([isinstance(a, vipy.activity.Activity) for a in activities]), "Invalid activity input; activities=[vipy.activity.Activity(), ...]"
            self._activities = {a.id():a for a in activities}

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
            strlist.append('fps=%1.1f' % float(self._framerate))
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
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
        assert isinstance(k, int), "Indexing video by frame must be integer"                
        if self.load().isloaded() and k >= 0 and k < len(self):
            dets = [t[k] for (tid,t) in self._tracks.items() if t[k] is not None]  # track interpolation (cloned) with boundary handling
            for d in dets:
                shortlabel = [(d.shortlabel(),'')]  # [(Noun, Verbing1), (Noun, Verbing2), ...]
                for (aid, a) in self._activities.items():  # insertion order:  First activity is primary, next is secondary
                    if a.hastrack(d.attributes['trackid']) and a.during(k):
                        # Shortlabel is always displayed as "Noun Verbing" during activity (e.g. Person Carrying, Vehicle Turning)
                        # If noun is associated with more than one activity, then this is shown as "Noun Verbing1\nNoun Verbing2", with a newline separator 
                        if not any([a.shortlabel() == v for (n,v) in shortlabel]):
                            shortlabel.append( (d.shortlabel(), a.shortlabel()) )
                        if 'activity' not in d.attributes:
                            d.attributes['activity'] = []                            
                        d.attributes['activity'].append(a)  # for activity correspondence (if desired)
                d.shortlabel( '\n'.join([('%s %s' % (n,v)).strip() for (n,v) in shortlabel[0 if len(shortlabel)==1 else 1:]]))
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
            self._currentframe = k    # used only for incremental add()
            yield self.__getitem__(k)
        self._currentframe = None

    def during(self, frameindex):
        try:
            self.__getitem__(frameindex)  # triggers load
            return True
        except:
            return False
        
    def frame(self, k):
        """Alias for self.__getitem__[k]"""
        return self.__getitem__(k)

    def frames(self):
        """Alias for __iter__()"""
        return self.__iter__()
    
    def labeled_frames(self):
        """Iterate over frames, yielding tuples (activity+object labelset in scene, vipy.image.Scene())"""
        self.load()
        for k in range(0, len(self)):
            self._currentframe = k    # used only for incremental add()
            yield (self.labels(k), self.__getitem__(k))
        self._currentframe = None
        
        
    def quicklook(self, n=9, dilate=1.5, mindim=256, fontsize=10, context=False):
        """Generate a montage of n uniformly spaced annotated frames centered on the union of the labeled boxes in the current frame to show the activity ocurring in this scene at a glance
           Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame.  This quicklook is most useful when len(self.activities()==1)
           for generating a quicklook from an activityclip().
        
           Input:
              -n:  Number of images in the quicklook
              -dilate:  The dilation factor for the bounding box prior to crop for display
              -mindim:  The minimum dimension of each of the elemnets in the montage
              -fontsize:  The size of the font for the bounding box label
              -context:  If true, replace the first and last frame in the montage with the full frame annotation, to help show the scale of the scene
        """
        if not self.isloaded():
            self.mindim(mindim).load()
        framelist = [int(np.round(f)) for f in np.linspace(0, len(self)-1, n)]
        imframes = [self.frame(k).maxmatte()  # letterbox or pillarbox
                    if (self.frame(k).boundingbox() is None) or (context is True and (k == framelist[0] or k == framelist[-1])) else
                    self.frame(k).padcrop(self.frame(k).boundingbox().dilate(dilate).imclipshape(self.width(), self.height()).maxsquare().int()).mindim(mindim, interp='nearest')
                    for k in framelist]  
        imframes = [im.savefig(fontsize=fontsize).rgb() for im in imframes]  # temp storage in memory
        return vipy.visualize.montage(imframes, imgwidth=mindim, imgheight=mindim)
    
    def tracks(self, tracks=None, id=None):
        """Return mutable dictionary of tracks"""        
        if tracks is None and id is None:
            return self._tracks  # mutable dict
        elif id is not None:
            return self._tracks[id]
        else:
            assert all([isinstance(t, vipy.object.Track) for t in tolist(tracks)]), "Invalid input - Must be vipy.object.Track or list of vipy.object.Track"
            self._tracks = {t.id():t for t in tolist(tracks)}  # insertion order preserved (python >=3.6)
            return self

    def tracklist(self):
        return list(self._tracks.values())
        
    def activities(self, activities=None, id=None):
        """Return mutable dictionary of activities.  All temporal alignment is relative to the current clip()."""
        if activities is None:
            return self._activities  # mutable dict
        elif id is not None:
            return self._activities[id]
        else:
            assert all([isinstance(a, vipy.activity.Activity) for a in tolist(activities)]), "Invalid input - Must be vipy.activity.Activity or list of vipy.activity.Activity"
            self._activities = {a.id():a for a in tolist(activities)}   # insertion order preserved (python >=3.6)
            return self

    def activitylist(self):
        return list(self._activities.values())  # insertion ordered (python >=3.6)
        
    def activityfilter(self, f):
        """Apply boolean lambda function f to each activity and keep activity if function is true, remove activity if function is false
        
           Usage:  Filter out all activities longer than 128 frames 
             vid = vid.activityfilter(lambda a: len(a)<128)

           Usage:  Filter out activities with category in set
             vid = vid.activityfilter(lambda a: a.category() in set(['category1', 'category2']))
       
        """
        self._activities = {k:a for (k,a) in self._activities.items() if f(a) == True}
        return self
        
    def trackfilter(self, f):
        """Apply lambda function f to each object and keep if filter is True"""
        self._tracks = {k:t for (k,t) in self._tracks.items() if f(t) == True} 
        return self

    def trackmap(self, f):
        """Apply lambda function f to each activity"""
        self._tracks = {k:f(t) for (k,t) in self._tracks.items()}
        assert all([isinstance(t, vipy.object.Track) for t in self.tracklist()]), "Lambda function must return vipy.object.Track()"
        return self
        
    def activitymap(self, f):
        """Apply lambda function f to each activity"""
        self._activities = {k:f(a) for (k,a) in self._activities.items()}
        assert all([isinstance(a, vipy.activity.Activity) for a in self.activitylist()]), "Lambda function must return vipy.activity.Activity()"
        return self

    def rekey(self):
        """Change the track and activity IDs to randomly assigned UUIDs.  Useful for cloning unique scenes"""
        d_old_to_new = {k:uuid.uuid1().hex for (k,a) in self._activities.items()}
        self._activities = {d_old_to_new[k]:a.id(d_old_to_new[k]) for (k,a) in self._activities.items()}
        d_old_to_new = {k:uuid.uuid1().hex for (k,t) in self._tracks.items()}
        self._tracks = {d_old_to_new[k]:t.id(d_old_to_new[k]) for (k,t) in self._tracks.items()}
        for (k,v) in d_old_to_new.items():
            self.activitymap(lambda a: a.replaceid(k,v) )
        return self
    
    def labels(self, k=None):
        """Return a set of all object and activity labels in this scene, or at frame int(k)"""
        return self.activitylabels(k).union(self.objectlabels(k))

    def activitylabels(self, startframe=None, endframe=None):
        """Return a set of all activity categories in this scene, or at startframe, or in range [startframe, endframe]"""        
        if startframe is None:
            return set([a.category() for a in self.activities().values()])
        elif startframe is not None and endframe is None:
            return set([a.category() for a in self.activities().values() if a.during(startframe)])
        elif startframe is not None and endframe is not None:
            return [set([a.category() for a in self.activities().values() if a.during(k)]) for k in range(startframe, endframe)] 
        else:
            raise ValueError('Invalid input - must specify both startframe and endframe, or only startframe')            
    
    def objectlabels(self, k=None):
        """Return a set of all activity categories in this scene, or at frame k"""
        return set([t.category() for t in self.tracks().values() if k is None or t.during(k)])        

    def categories(self):
        """Alias for labels()"""
        return self.labels()
    
    def activity_categories(self):
        """Alias for activitylabels()"""
        return self.activitylabels()        
        
    def hasactivities(self):
        return len(self._activities) > 0

    def hastracks(self):
        return len(self._tracks) > 0

    def hastrack(self, trackid):
        return trackid in self._tracks

    def add(self, obj, category=None, attributes=None, rangecheck=True):
        """Add the object obj to the scene, and return an index to this object for future updates
        
        This function is used to incrementally build up a scene frame by frame.  Obj can be one of the following types:

            * obj = vipy.object.Detection(), this must be called from within a frame iterator (e.g. for im in video) to get the current frame index
            * obj = vipy.object.Track()  
            * obj = vipy.activity.Activity()
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
            if rangecheck and not obj.hasoverlap(width=self.width(), height=self.height()):
                raise ValueError("Track '%s' does not intersect with frame shape (%d, %d)" % (str(t), self.height(), self.width()))
            self._tracks[t.id()] = t
            return t.id()
        elif isinstance(obj, vipy.object.Track):
            if rangecheck and not vipy.geometry.imagebox(self.shape()).inside(obj.boundingbox()):
                obj = obj.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('Clipping track "%s" to image rectangle' % (str(obj)))
            self._tracks[obj.id()] = obj
            return obj.id()
        elif isinstance(obj, vipy.activity.Activity):
            if rangecheck and obj.startframe() >= obj.endframe():
                raise ValueError("Activity '%s' has invalid (startframe, endframe)=(%d, %d)" % (str(obj), obj.startframe(), obj.endframe()))
            self._activities[obj.id()] = obj
            # FIXME: check to see if activity has at least one track during activity
            return obj.id()
        elif (istuple(obj) or islist(obj)) and len(obj) == 4 and isnumber(obj[0]):
            assert self._currentframe is not None, "add() for obj=xywh must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=category, keyframes=[self._currentframe], boxes=[vipy.geometry.BoundingBox(xywh=obj)], boundary='strict', attributes=attributes)
            if rangecheck and not vipy.geometry.imagebox(self.shape()).inside(t.boundingbox()):
                t = t.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('Clipping track "%s" to image rectangle' % (str(t)))
            self._tracks[t.id()] = t
            return t.id()
        else:
            raise ValueError('Undefined object type "%s" to be added to scene - Supported types are obj in ["vipy.object.Detection", "vipy.object.Track", "vipy.activity.Activity", "[xmin, ymin, width, height]"]' % str(type(obj)))        

    def clear(self):
        """Remove all activities and tracks from this object"""
        self._activities = {}
        self._tracks = {}
        return self
        
    def dict(self):
        d = super(Scene, self).dict()
        d['category'] = self.category()
        d['tracks'] = [t.dict() for t in self._tracks.values()]
        d['activities'] = [a.dict() for a in self._activities.values()]
        return d
        
    def csv(self, outfile=None):
        """Export scene to CSV file format with header.  If there are no tracks, this will be empty. """
        assert self.load().isloaded()
        csv = [(k,  # frame number (zero indexed)
                d.category(), d.shortlabel(), # track category and shortlabel (displayed in caption)
                ';'.join([a.category() for a in d.attributes['activity']] if 'activity' in d.attributes else ''), # semicolon separated activity ID assocated with track
                d.xmin(), d.ymin(), d.width(), d.height(),   # bounding box
                d.attributes['trackid'],  # globally unique track ID
                ';'.join([a.id() for a in d.attributes['activity']] if 'activity' in d.attributes else '')) # semicolon separated activity ID assocated with track
               for (k,im) in enumerate(self) for d in im.objects()]
        csv = [('# frame number', 'object category', 'object shortlabel', 'activity categories(;)', 'xmin', 'ymin', 'width', 'height', 'trackid', 'activity id(;)')] + csv
        return writecsv(csv, outfile) if outfile is not None else csv


    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations"""
        if fps is None:
            return self._framerate
        else:
            assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"        
            self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
            self._tracks = {k:t.framerate(fps) for (k,t) in self._tracks.items()}
            self._activities = {k:a.framerate(fps) for (k,a) in self._activities.items()}        
            self._framerate = fps
            return self
        
    def activitysplit(self):
        """Split the scene into k separate scenes, one for each activity.  Do not include overlapping activities.  This is useful for union()"""
        vid = self.clone(flushforward=True)
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))
        activities = sorted([a.clone() for a in vid.activities().values() if (a.endframe()-a.startframe()) > 0], key=lambda a: a.startframe())   # only activities with at least one frame, sorted in temporal order
        tracks = [ [t.clone() for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in activities]  # tracks associated with each activity (may be empty)
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        return [vid.clone().activities(pa).tracks(t) for (pa,t) in zip(activities, tracks)]

    def activityclip(self, padframes=0, multilabel=True):
        """Return a list of vipy.video.Scene() each clipped to be temporally centered on a single activity, with an optional padframes before and after.  
           The Scene() category is updated to be the activity, and only the objects participating in the activity are included.
           Activities are returned ordered in the temporal order they appear in the video.
           The returned vipy.video.Scene() objects for each activityclip are clones of the video, with the video buffer flushed.
           Each activityclip() is associated with each activity in the scene, and includes all other secondary activities that the objects in the primary activity also perform (if multilabel=True).  See activityclip().labels(). 
           Calling activityclip() on activityclip(multilabel=True) can result in duplicate activities, due to the overlapping secondary activities being included in each clip.  Be careful. 
        """
        vid = self.clone(flushforward=True)
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity clips with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))
        primary_activities = sorted([a.clone() for a in vid.activities().values() if (a.endframe()-a.startframe()) > 0], key=lambda a: a.startframe())   # only activities with at least one frame, sorted in temporal order
        tracks = [ [t.clone() for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in primary_activities]  # tracks associated with each primary activity (may be empty)
        secondary_activities = [[sa.clone() for sa in primary_activities if (sa.id() != pa.id() and pa.temporal_iou(sa)>0 and (len(T)==0 or any([sa.hastrack(t) for t in T])))] for (pa, T) in zip(primary_activities, tracks)]  # overlapping secondary activities that includes any track in the primary activity
        secondary_activities = [sa if multilabel else [] for sa in secondary_activities]  
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        padframes = padframes if istuple(padframes) else (padframes,padframes)
        return [vid.clone().activities([pa]+sa).tracks(t).clip(startframe=max(pa.startframe()-padframes[0], 0),
                                                               endframe=(pa.endframe()+padframes[1])).category(pa.category()) for (pa,sa,t) in zip(primary_activities, secondary_activities, tracks)]

    def trackbox(self, dilate=1.0):
        """The trackbox is the union of all track bounding boxes in the video, or the image rectangle if there are no tracks"""
        boxes = [t.boundingbox().dilate(dilate) for t in self.tracklist()]
        return boxes[0].union(boxes[1:]) if len(boxes) > 0 else imagebox(self.shape())
        
    def activitybox(self, activityid=None, dilate=1.0):
        """The activitybox is the union of all activity bounding boxes in the video, which is the union of all tracks contributing to all activities.  This is most useful after activityclip().
           The activitybox is the smallest bounding box that contains all of the boxes from all of the tracks in all activities in this video.
        """
        activities = [a for (k,a) in self.activities().items() if (activityid is None or k in set(activityid))]
        boxes = [t.boundingbox().dilate(dilate) for t in self.tracklist() if any([a.hastrack(t) for a in activities])]
        return boxes[0].union(boxes[1:]) if len(boxes) > 0 else vipy.geometry.BoundingBox(xmin=0, ymin=0, width=self.width(), height=self.height())

    def activitycuboid(self, activityid=None, dilate=1.0, maxdim=256, bb=None):
        """The activitycuboid() is the fixed square spatial crop corresponding to the activitybox (or supplied bounding box), which contains all of the valid activities in the scene.  This is most useful after activityclip().
           The activitycuboid() is a spatial crop of the video corresponding to the supplied boundingbox or the square activitybox().
           This crop must be resized such that the maximum dimension is provided since the crop can be tiny and will not be encodable by ffmpeg
        """
        bb = self.activitybox(activityid).maxsquare() if bb is None else bb  
        assert bb is None or isinstance(bb, vipy.geometry.BoundingBox)
        assert bb.issquare(), "Add support for non-square boxes"
        return self.clone().crop(bb.dilate(dilate).int(), zeropad=True).resize(maxdim, maxdim)  # crop triggers preview()

    def activitysquare(self, activityid=None, dilate=1.0, maxdim=256):
        """The activity square is the maxsquare activitybox that contains only valid (non-padded) pixels interior to the image"""
        bb = self.activitybox(activityid).maxsquare().dilate(dilate).int().iminterior(self.width(), self.height()).minsquare()
        return self.activitycuboid(activityid, dilate=1.0, maxdim=maxdim, bb=bb)

    def activitytube(self, activityid=None, dilate=1.0, maxdim=256):
        """The activitytube() is a sequence of crops where the spatial box changes on every frame to track the activity.  
           The box in each frame is the square activitybox() for this video which is the union of boxes contributing to this activity in each frame.
           This function does not perform any temporal clipping.  Use activityclip() first to split into individual activities.  
           Crops will be optionally dilated, with zeropadding if the box is outside the image rectangle.  All crops will be resized so that the maximum dimension is maxdim (and square by default)
        """
        vid = self.clone().load()  # triggers load
        self.activityfilter(lambda a: activityid is None or a.id() in set(activityid))  # only requested IDs (or all of them)
        frames = [im.padcrop(im.boundingbox().maxsquare().dilate(dilate).int()).resize(maxdim, maxdim) for im in vid if im.boundingbox() is not None]  # track interpolation, for frames with boxes only
        if len(frames) != len(vid):
            warnings.warn('[vipy.video.activitytube]: Removed %d frames with no spatial bounding boxes' % (len(vid) - len(frames)))
            vid.attributes['activtytube'] = {'truncated':len(vid) - len(frames)}  # provenance to reject
        if len(frames) == 0:
            warnings.warn('[vipy.video.activitytube]: Resulting video is empty!  Setting activitytube to zero')
            frames = [ vid[0].resize(maxdim, maxdim).zeros() ]  # empty frame
            vid.attributes['activitytube'] = {'empty':True}   # provenance to reject 
        vid._tracks = {ti:vipy.object.Track(keyframes=[f for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],
                                            boxes=[d for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],
                                            category=t.category(), trackid=ti)
                       for (k,(ti,t)) in enumerate(self._tracks.items())}  # replace tracks with boxes relative to tube
        return vid.array(np.stack([im.numpy() for im in frames]))

    def actortube(self, trackid, dilate=1.0, maxdim=256):
        """The actortube() is a sequence of crops where the spatial box changes on every frame to track the primary actor performing an activity.  
           The box in each frame is the square box centered on the primary actor performing the activity, dilated by a given factor (the original box around the actor is unchanged, this just increases the context, with zero padding)
           This function does not perform any temporal clipping.  Use activityclip() first to split into individual activities.  
           All crops will be resized so that the maximum dimension is maxdim (and square by default)
        """
        assert self.hastrack(trackid), "Track ID %s not found - Actortube requires a track ID in the scene (tracks=%s)" % (str(trackid), str(self.tracks()))
        vid = self.clone().load()  # triggers load        
        t = vid.tracks(id=trackid)  # actor track
        frames = [im.padcrop(t[k].maxsquare().dilate(dilate).int()).resize(maxdim, maxdim) for (k,im) in enumerate(vid) if t.during(k)]  # track interpolation, for frames with boxes for this actor only
        if len(frames) != len(vid):
            warnings.warn('[vipy.video.actortube]: Removed %d frames with no spatial bounding boxes for actorid "%s"' % (len(vid) - len(frames), trackid))
            vid.attributes['actortube'] = {'truncated':len(vid) - len(frames)}  # provenance to reject
        if len(frames) == 0:
            warnings.warn('[vipy.video.actortube]: Resulting video is empty!  Setting actortube to zero')
            frames = [ vid[0].resize(maxdim, maxdim).zeros() ]  # empty frame
            vid.attributes['actortube'] = {'empty':True}   # provenance to reject 
        vid._tracks = {ti:vipy.object.Track(keyframes=[f for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],
                                            boxes=[d for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],
                                            category=t.category(), trackid=ti)  # preserve trackid
                       for (k,(ti,t)) in enumerate(self._tracks.items())}  # replace tracks with boxes relative to tube
        return vid.array(np.stack([im.numpy() for im in frames]))


    def clip(self, startframe, endframe):
        """Clip the video to between (startframe, endframe).  This clip is relative to clip() shown by __repr__().  Return a clone of the video for idemponence"""
        v = super(Scene, self.clone()).clip(startframe, endframe)  # clone for idemponence
        v._tracks = {k:t.offset(dt=-startframe) for (k,t) in v._tracks.items()}   # track offset is performed here, not within activity
        v._activities = {k:a.offset(dt=-startframe) for (k,a) in v._activities.items()}        
        return v  

    def cliptime(self, startsec, endsec):
        raise NotImplementedError('FIXME: use clip() instead for now')
            
    def crop(self, bb, zeropad=True):
        """Crop the video using the supplied box, update tracks relative to crop, video is zeropadded if box is outside frame rectangle"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        bb = bb.int()
        if zeropad and bb != bb.clone().imclipshape(self.width(), self.height()):
            self.zeropad(bb.width(), bb.height())     
            bb = bb.offset(bb.width(), bb.height())            
        super(Scene, self).crop(bb, zeropad=False)  # range check handled here to correctly apply zeropad
        self._tracks = {k:t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for (k,t) in self._tracks.items()}
        return self
    
    def zeropad(self, padwidth, padheight):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        assert isinstance(padwidth, int) and isinstance(padheight, int)
        super(Scene, self).zeropad(padwidth, padheight)  
        self._tracks = {k:t.offset(dx=padwidth, dy=padheight) for (k,t) in self._tracks.items()}
        return self
        
    def fliplr(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.fliplr(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).fliplr()
        return self

    def flipud(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.flipud(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).flipud()
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90ccw(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).rot90ccw()
        return self

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90cw(H,W) for (k,t) in self._tracks.items()}
        super(Scene, self).rot90cw()
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self._tracks = {k:t.scalex(sx) for (k,t) in self._tracks.items()}
        self._tracks = {k:t.scaley(sy) for (k,t) in self._tracks.items()}
        super(Scene, self).resize(rows, cols)
        return self

    def mindim(self, dim):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return self.resize(cols=dim) if W>H else self.resize(rows=dim)
    
    def rescale(self, s):
        """Spatially rescale the scene by a constant scale factor"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        self._tracks = {k:t.rescale(s) for (k,t) in self._tracks.items()}
        super(Scene, self).rescale(s)
        return self

    def union(self, other, temporal_iou_threshold=0.5, spatial_iou_threshold=0.9, strict=True):
        """Compute the union two scenes as the set of unique activities.  

           A pair of activities or tracks are non-unique if they overlap spatially and temporally by a given IoU threshold.  Merge overlapping tracks. 
  
           Input:
             -Other: Scene or list of scenes for union
             -spatial_iou_threshold:  The intersection over union threshold for an activity bounding box (the union of all tracks within the activity) to be declared duplicates.  Disable by setting to 1.0
             -temporal_iou_threshold:  The intersection over union threshold for a temporal bounding box for a pair of activities to be declared duplicates.  Disable by setting to 1.0
             -strict:  Require both scenes to share the same underlying video filename

           Output:
             -Updates this scene to include the non-overlapping activities from other.  By default, it takes the strict union of all activities and tracks. 
        """
        for o in tolist(other):
            assert isinstance(o, Scene), "Invalid input - must be vipy.video.Scene() object and not type=%s" % str(type(o))
            assert spatial_iou_threshold >= 0 and spatial_iou_threshold <= 1, "invalid spatial_iou_threshold, must be between [0,1]"
            assert temporal_iou_threshold >= 0 and temporal_iou_threshold <= 1, "invalid temporal_iou_threshold, must be between [0,1]"        
            if strict:
                assert self.filename() == o.filename(), "Invalid input - Scenes must have the same underlying video.  Disable this with strict=False."
            otherclone = o.clone()   # do not change other, make a copy

            # Merge tracks 
            for (i,ti) in self.tracks().items():
                for (j,tj) in otherclone.tracks().items():
                    if ti.category() == tj.category() and ti.endpointiou(tj) > spatial_iou_threshold:  # maximum framewise overlap at endpoints >threshold
                        print('[vipy.video.union]: merging track "%s" -> "%s" for scene "%s"' % (str(ti), str(tj), str(self)))
                        self.tracks()[i] = ti.average(tj)  # merge duplicate tracks
                        otherclone = otherclone.activitymap(lambda a: a.replace(tj, ti))  # replace duplicate track reference in activity
                        otherclone = otherclone.trackfilter(lambda t: t.id() != j)  # remove duplicate track
            
            # Dedupe activities
            for (i,ai) in self.activities().items():
                for (j,aj) in otherclone.activities().items():
                    if ai.category() == aj.category() and set(ai.trackids()) == set(aj.trackids()) and ai.temporal_iou(aj) > temporal_iou_threshold:
                        otherclone = otherclone.activityfilter(lambda a: a.id() != j)  # remove duplicate activity

            # Union of unique tracks/activities
            if len(set(self.tracks().keys()).intersection(otherclone.tracks().keys())) > 0:
                print('[vipy.video.union]: track key collision - Ignoring key from other')
            if len(set(self.activities().keys()).intersection(otherclone.activities().keys())) > 0:
                print('[vipy.video.union]: activity key collision - Ignoring key from other')                
            self.tracks().update(otherclone.tracks())
            self.activities().update(otherclone.activities())

        return self        

    def annotate(self, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[]):
        """Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, pixels in this video will now contain the overlay
        This function does not play the video, it only generates an annotation video frames.  Use show() which is equivalent to annotate().saveas().play()
        In general, this function should not be run on very long videos, as it requires loading the video framewise into memory, try running on clips instead.
        """
        if verbose and not self.isloaded():
            print('[vipy.video.annotate]: Loading video ...')  
        
        assert self.load().isloaded(), "Load() failed"
        
        if verbose:
            print('[vipy.video.annotate]: Annotating video ...')              
        imgs = [self[k].savefig(fontsize=fontsize,
                                captionoffset=captionoffset,
                                textfacecolor=textfacecolor,
                                textfacealpha=textfacealpha,
                                shortlabel=shortlabel,
                                boxalpha=boxalpha,
                                d_category2color=d_category2color,
                                categories=categories,
                                nocaption=nocaption,
                                nocaption_withstring=nocaption_withstring).numpy() for k in range(0, len(self))]  # SLOW for large videos
        self._array = np.stack([np.array(PIL.Image.fromarray(img).convert('RGB')) for img in imgs], axis=0)  # replace pixels with annotated pixels
        return self


    def show(self, outfile=None, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[]):
        """Generate an annotation video saved to outfile (or tempfile if outfile=None) and show it using ffplay when it is done exporting.  Do not modify the original video buffer"""
        return self.clone().annotate(verbose=verbose, 
                                     fontsize=fontsize,
                                     captionoffset=captionoffset,
                                     textfacecolor=textfacecolor,
                                     textfacealpha=textfacealpha,
                                     shortlabel=shortlabel,
                                     boxalpha=boxalpha,
                                     d_category2color=d_category2color,
                                     categories=categories,
                                     nocaption=nocaption, 
                                     nocaption_withstring=nocaption_withstring).saveas(outfile).play()
    
    def thumbnail(self, outfile=None, frame=0, fontsize=10, nocaption=False, boxalpha=0.25, dpi=200, textfacecolor='white', textfacealpha=1.0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile"""
        return self.__getitem__(frame).savefig(outfile if outfile is not None else temppng(), fontsize=fontsize, nocaption=nocaption, boxalpha=boxalpha, dpi=dpi, textfacecolor=textfacecolor, textfacealpha=textfacealpha)

    
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
                                keyframes=[0, np.random.randint(50,100), 150],
                                boxes=[vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.geometry.BoundingBox(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,32)]

    activities = [vipy.activity.Activity(label='activity%d' % k, shortlabel='a%d' % k, tracks=[tracks[j].id() for j in [np.random.randint(32)]], startframe=np.random.randint(50,99), endframe=np.random.randint(100,150)) for k in range(0,32)]   
    return Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)


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

    activities = [vipy.activity.Activity(label='Person Carrying', shortlabel='Carry', tracks=[tracks[0].id(), tracks[1].id()], startframe=np.random.randint(20,50), endframe=np.random.randint(70,100))]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)

    return ims
    

