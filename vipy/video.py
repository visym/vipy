import os
import sys
import dill
from vipy.globals import print
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng, \
    istuple, islist, isnumber, tolist, filefull, fileext, isS3url, totempdir, flatlist, tocache, premkdir, writecsv, iswebp, ispng, isgif, filepath, Stopwatch, toextension, isjsonfile, isRTSPurl, isRTMPurl
from vipy.image import Image
import vipy.geometry
import vipy.math
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
import time
from io import BytesIO
import itertools
import vipy.globals
import vipy.activity
import hashlib
from pathlib import PurePath
import queue 
import threading
from concurrent.futures import ThreadPoolExecutor


try:
    import ujson as json  # faster
except ImportError:
    import json
    
    
ffmpeg_exe = shutil.which('ffmpeg')
has_ffmpeg = ffmpeg_exe is not None and os.path.exists(ffmpeg_exe)
ffprobe_exe = shutil.which('ffprobe')        
has_ffprobe = ffprobe_exe is not None and os.path.exists(ffprobe_exe)
ffplay_exe = shutil.which('ffplay')        
has_ffplay = ffplay_exe is not None and os.path.exists(ffplay_exe)


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

    >>> vid = vipy.video.Video(url='s3://BUCKET.s3.amazonaws.com/PATH/video.ext')

    If you set the environment variables VIPY_AWS_ACCESS_KEY_ID and VIPY_AWS_SECRET_ACCESS_KEY, then this will download videos directly from S3 using boto3 and store in VIPY_CACHE.
    Note that the URL protocol should be 's3' and not 'http' to enable keyed downloads.  

    >>> vid = vipy.video.Video(array=array, colorspace='rgb')
    
    The input 'array' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video
    Note that some video transformations are only available prior to load(), and the array() is assumed immutable after load().

    >>> frames = [im for im in vipy.video.RandomVideo()]
    >>> vid = vipy.video.Video(frames=frames)

    The input can be an RTSP video stream.  Note that streaming is most efficiently performed using `vipy.video.Scene`.  The URL must contain the 'rtsp://' url scheme.  
    You can experiment with this using the free Periscope H.264 RTSP App (https://apps.apple.com/us/app/periscope-hd-h-264-rtsp-cam/id1095600218)

    >>> vipy.video.Scene(url='rtsp://127.0.0.1:8554/live.sdp').show()
    >>> for im in vipy.video.Scene(url='rtsp://127.0.0.1:8554/live.sdp').stream():
    >>>     print(im)

    Args:
        filename: [str] The path to a video file.  
        url: [str] The URL to a video file.  If filename is not provided, then a random filename is assigned in VIPY_CACHE on download
        framerate: [float] The framerate of the video file.  This is required.  You can introspect this using ffprobe.
        attributes: [dict]  A user supplied dictionary of metadata about this video.
        colorspace: [str] Must be in ['rgb', 'float']
        array: [numpy] An NxHxWxC numpy array for N frames each HxWxC shape
        startframe: [int]  A start frame to clip the video
        endframe: [int] An end frame to clip the video
        startsec: [float] A start time in seconds to clip the video (this requires setting framerate)
        endsec: [float] An end time in seconds to clip the video (this requires setting framerate)
        frames: [list of `vipy.image.Image`] A list of frames in the video
        probeshape: [bool] If true, then probe the shape of the video from ffprobe to avoid an explicit preview later.  This can speed up loading in some circumstances.

    """
    def __init__(self, filename=None, url=None, framerate=30.0, attributes=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None, frames=None, probeshape=False):
        self._url = None
        self._filename = None
        self._array = None
        self._colorspace = None
        self._ffmpeg = None
        self._framerate = None

        self.attributes = attributes if attributes is not None else {}
        assert isinstance(self.attributes, dict), "Attributes must be a python dictionary"
        assert filename is not None or url is not None or array is not None or frames is not None, 'Invalid constructor - Requires "filename", "url" or "array" or "frames"'

        # FFMPEG installed?
        if not has_ffmpeg:
            warnings.warn('"ffmpeg" executable not found on path, this is required for vipy.video - Install from http://ffmpeg.org/download.html')

        # Constructor clips
        startframe = startframe if startframe is not None else (0 if endframe is not None else startframe)
        assert (startsec is not None and endsec is not None) or (startsec is None and endsec is None), "Invalid input - (startsec,endsec) are both required"        
        (self._startframe, self._endframe) = (None, None)  # __repr__ only
        (self._startsec, self._endsec) = (None, None)  # __repr__ only (legacy, no longer used)

        # Input filenames
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
            self._url = url
        if filename is not None:
            self._filename = os.path.normpath(os.path.expanduser(filename))
        elif self._url is not None:
            if isS3url(self._url):
                self._filename = totempdir(self._url)  # Preserve S3 Object ID
            elif isRTSPurl(self._url) or isRTMPurl(self._url):
                # https://ffmpeg.org/ffmpeg-protocols.html#rtsp                
                self._filename = self._url                
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            elif isyoutubeurl(self._url):
                self._filename = os.path.join(tempdir(), '%s' % self._url.split('?')[1].split('&')[0])
            else:
                self._filename = totempdir(self._url)  
            if vipy.globals.cache() is not None and self._filename is not None and not isRTSPurl(self._filename) and not isRTMPurl(self._filename):
                self._filename = os.path.join(remkdir(vipy.globals.cache()), filetail(self._filename))

        # Initial video shape: useful to avoid preview()
        self._ffmpeg = ffmpeg.input(self.filename())  # restore, no other filters        
        if probeshape and (frames is None and array is None) and has_ffprobe and self.hasfilename():
            self.shape(self.probeshape())
        else:
            self._shape = None  # preview() on shape()
            
        # Video filter chain
        if framerate is not None:
            if array is None and frames is None:
                self.framerate(framerate)
            self._framerate = framerate        
        if startframe is not None:
            self.clip(startframe, endframe)  
        if startsec is not None:
            # WARNING: if the user does not supply the correct framerate for the video, then this will be wrong since these are converted to frames 
            self.clip(int(round(startsec/self.framerate())), int(round(endsec/self.framerate())) if endsec is not None else None)

            
        # Array input
        assert not (array is not None and frames is not None)
        if array is not None:
            self.array(array)
            self.colorspace(colorspace)
        elif frames is not None and (isinstance(frames, list) or isinstance(frames, tuple)) and all([isinstance(im, vipy.image.Image) for im in frames]):
            self.fromframes(frames)
        elif frames is not None and (isinstance(frames, list) or isinstance(frames, tuple)) and all([isinstance(im, str) and os.path.exists(im) for im in frames]):
            self.fromframes([vipy.image.Image(filename=f) for f in frames])
        elif frames is not None and (isinstance(frames, str) and os.path.isdir(frames)):
            self.fromdirectory(frames)
            
        
    @classmethod
    def cast(cls, v):
        """Cast a conformal video object to a `vipy.video.Video` object.
        
        This is useful for downcasting superclasses.

        >>> vs = vipy.video.RandomScene()
        >>> v = vipy.video.Video.cast(vs)

        """
        assert isinstance(v, vipy.video.Video), "Invalid input - must be derived from vipy.video.Video"
        v.__class__ = vipy.video.Video
        return v
            
    @classmethod
    def from_json(cls, s):
        """Import a json string as a `vipy.video.Video` object.

        This will perform a round trip from a video to json and back to a video object.
        This same operation is used for serialization of all vipy objects to JSON for storage.

        >>> v = vipy.video.Video.from_json(vipy.video.RandomVideo().json())

        """
        
        d = json.loads(s) if not isinstance(s, dict) else s
        v = cls(filename=d['_filename'],
                url=d['_url'],
                framerate=d['_framerate'],
                array=np.array(d['_array']) if d['_array'] is not None else None,
                colorspace=d['_colorspace'],
                attributes=d['attributes'],
                startframe=d['_startframe'],
                endframe=d['_endframe'],
                startsec=d['_startsec'],
                endsec=d['_endsec'])
        v._ffmpeg = v._from_ffmpeg_commandline(d['_ffmpeg'])
        return v.filename(d['_filename']) if d['_filename'] is not None else v.nofilename()
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, color=%s" % (self.height(), self.width(), len(self), self.colorspace()))
        if self.filename() is not None:
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if not self.isloaded() and self._startframe is not None and self._endframe is None:
            strlist.append('clip=(%d,)' % (self._startframe))
        if self._framerate is not None:
            strlist.append('fps=%1.1f' % float(self._framerate))
        return str('<vipy.video: %s>' % (', '.join(strlist)))

    def __len__(self):
        """Number of frames in the video if loaded, else zero.  
        
        .. notes:: Do not automatically trigger a load, since this can interact in unexpected ways with other tools that depend on fast __len__()
        """
        if not self.isloaded():
            warnings.warn('Load() video to see number of frames - Returning zero')  # should this just throw an exception?
        return len(self.array()) if self.isloaded() else 0

    def __getitem__(self, k):
        """Alias for `vipy.video.Video.frame`"""
        return self.frame(k)

    def metadata(self):
        """Return a dictionary of metadata about this video.

        This is an alias for the 'attributes' dictionary. 
        """
        return self.attributes

    def videoid(self, newid=None):
        """Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by SHA1 hash of the `vipy.video.Video.filename` and `vipy.video.Video.url`.

        Args:
            newid: [str] If not None, then update the video_id as newid. 

        Returns:
            The video ID if newid=None else self

        .. note::
            - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change.
            - If a video does not have a filename or URL or a video ID in the attributes, then this will return None
            - To preserve a video ID independent of transformations, set self.setattribute('video_id', ${MY_ID}), or pass in newid
        """
        if newid is not None:
            self.setattribute('video_id', newid)
            return self
        else:
            return self.attributes['video_id'] if 'video_id' in self.attributes else (hashlib.sha1(str(str(self.filename())+str(self.url())).encode("UTF-8")).hexdigest() if (self.filename() is not None or self.url() is not None) else None)
        

    def frame(self, k, img=None):
        """Return the kth frame as an `vipy.image Image` object"""        
        assert isinstance(k, int) and k>=0, "Frame index must be non-negative integer"
        return Image(array=img if img is not None else (self._array[k] if self.isloaded() else self.preview(k).array()), colorspace=self.colorspace())       
        
    def __iter__(self):
        """Iterate over loaded video, yielding mutable frames.
        
        FIXME: this should really be replaced by default with:
        
        >>> for in im self.stream():
        >>>     pass

        This is much more efficient, and will be migrated to the default for frame iterators.

        """
        self.load().numpy()  # triggers load and copy of read-only video buffer, mutable
        with np.nditer(self._array, op_flags=['readwrite']) as it:
            for k in range(0, len(self)):
                self._currentframe = k    # used only for incremental add()                
                yield self.__getitem__(k)
            self._currentframe = None    # used only for incremental add()

    def store(self):
        """Store the current video file as an attribute of this object.  

        Useful for archiving an object to be fully self contained without any external references.  

        >>> v == v.store().restore(v.filename()) 
        
        .. note::
        -Remove this stored video using unstore()
        -Unpack this stored video and set up the video chains using restore() 
        -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string.
        -Useful for creating a single self contained object for distributed processing.  
        """
        assert self.hasfilename(), "Video file not found"
        with open(self.filename(), 'rb') as f:
            self.attributes['__video__'] = f.read()
        return self

    def unstore(self):
        """Delete the currently stored video from `vipy.video.Video.store"""
        return self.delattribute('__video__')

    def restore(self, filename):
        """Save the currently stored video as set using `vipy.video.Video.store` to filename, and set up filename"""
        assert self.hasattribute('__video__'), "Video not stored"
        with open(filename, 'wb') as f:
            f.write(self.attributes['__video__'])
        return self.filename(filename)                
        
    def stream(self, write=False, overwrite=False, bufsize=1024):
        """Iterator to yield groups of frames streaming from video.

        A video stream is a real time iterator to read or write from a video.  Streams are useful to group together frames into clips that are operated on as a group.

        The following use cases are supported:

        >>> v = vipy.video.RandomScene()

        Stream individual video frames lagged by 10 frames and 20 frames

        >>> for (im1, im2) in zip(v.stream().frame(n=-10), v.stream().frame(n=-20)):
        >>>     print(im1, im2)
        
        Stream overlapping clips such that each clip is a video n=16 frames long and starts at frame i, and the next clip is n=16 frames long and starts at frame i=i+m

        >>> for vc in v.stream().clip(n=16, m=4):
        >>>     print(vc)

        Stream non-overlapping batches of frames such that each clip is a video of length n and starts at frame i, and the next clip is length n and starts at frame i+n

        >>> for vb in v.stream().batch(n=16):
        >>>     print(vb)        

        Create a write stream to incrementally add frames to long video.  

        >>> vi = vipy.video.Video(filename='/path/to/output.mp4')
        >>> vo = vipy.video.Video(filename='/path/to/input.mp4')
        >>> with vo.stream(write=True) as s:
        >>>     for im in vi.stream():
        >>>         s.write(im)  # manipulate pixels of im, if desired

        Create a YouTube live stream from an RTSP camera at 5Hz 
        
        >>> vo = vipy.video.Scene(url='rtmp://a.rtmp.youtube.com/live2/$SECRET_STREAM_KEY')
        >>> vi = vipy.video.Scene(url='rtsp://URL').framerate(5)
        >>> with vo.framerate(5).stream(write=True) as s:
        >>>     for im in vi.framerate(5).stream():
        >>>         s.write(im.mindim(256))

        Args:
            write: [bool]  If true, create a write stream
            overwrite: [bool]  If true, and the video output filename already exists, overwrite it
            bufsize: [int]  The maximum queue size for the pipe thread.  

        Returns:
            A Stream object

        ..note:: Using this iterator may affect PDB debugging due to stdout/stdin redirection.  Use ipdb instead.

        """

        class Stream(object):
            def __init__(self, v, bufsize=bufsize):
                self._video = v   # do not clone
                self._write_pipe = None
                self._frame_index = 0
                self._vcodec = 'libx264'
                self._framerate = self._video.framerate()
                self._outfile = self._video.filename()
                self._write = write or overwrite               
                assert self._write is False or (overwrite is True or not os.path.exists(self._outfile)), "Output file '%s' exists - Writable stream cannot overwrite existing video file unless overwrite=True" % self._outfile
                if overwrite and os.path.exists(self._outfile):
                    os.remove(self._outfile)                
                self._shape = self._video.shape() if (not self._write) or (self._write and self._video.canload()) else None  # shape for write can be defined by first frame
                assert (write is True or overwrite is True) or self._shape is not None, "Invalid video '%s'" % (str(v))
                self._queuesize = bufsize
                
            def __enter__(self):
                """Write pipe context manager"""
                assert self._write, "invalid parameters for write only context manager"

                if self._shape is not None:
                    (height, width) = self._shape
                    outfile = self._outfile if self._outfile is not None else self._url  # may be youtube/twitch live stream
                    outrate = 30 if vipy.util.isRTMPurl(outfile) else self._video.framerate()
                    fiv = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=self._video.framerate()) 
                           .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2'))
                    fi = ffmpeg.concat(fiv.filter('fps', fps=30, round='up'), ffmpeg.input('anullsrc', f='lavfi'), v=1, a=1) if isRTMPurl(outfile) else fiv  # empty audio for youtube-live
                    fo = (fi.output(filename=self._outfile if self._outfile is not None else self._url, pix_fmt='yuv420p', vcodec=self._vcodec, f='flv' if vipy.util.isRTMPurl(outfile) else vipy.util.fileext(outfile, withdot=False), g=2*outrate)
                          .overwrite_output() 
                          .global_args('-cpuflags', '0', '-loglevel', 'quiet' if not vipy.globals.isdebug() else 'debug'))
                    self._write_pipe = fo.run_async(pipe_stdin=True)
                    
                self._frame_index = 0
                return self
            
            def __exit__(self, type, value, tb):
                """Write pipe context manager
                
                ..note:: This is triggered on ctrl-c as the last step for cleanup
                """
                if self._write_pipe is not None:
                    self._write_pipe.stdin.close()
                    self._write_pipe.wait()
                    del self._write_pipe
                    self._write_pipe = None
                if type is not None:
                    raise
                return self
                
            def __call__(self, im):
                """alias for write()"""
                return self.write(im)

            def _read_pipe(self):
                if not self._video.isloaded():
                    p = self._video._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24').global_args('-nostdin', '-loglevel', 'debug' if vipy.globals.isdebug() else 'quiet').run_async(pipe_stdout=True, pipe_stderr=True)
                    assert p is not None, "Invalid read pipe"
                    p.poll()
                    return p
                else:
                    return None
            
            def __iter__(self):
                """Stream individual video frames"""
                
                if self._video.isloaded():
                    # For loaded video, just use the existing iterator for in-memory video
                    for im in self._video.__iter__():
                        yield im
                else:
                    p = self._read_pipe()
                    q = queue.Queue(self._queuesize)
                    (h, w) = self._shape
                    
                    def _f_threadloop(pipe, queue, height, width, event):
                        assert pipe is not None, "Invalid pipe"
                        assert queue is not None, "invalid queue"
                        while True:
                            in_bytes = pipe.stdout.read(height * width * 3)
                            if not in_bytes:
                                queue.put(None)
                                pipe.poll()
                                pipe.wait()
                                if pipe.returncode != 0:
                                    raise ValueError('Stream iterator failed with returncode %d - error "%s"' % (pipe.returncode, str(pipe.stderr.readlines())))
                                event.wait()
                                break
                            else:
                                queue.put(np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))

                    e = threading.Event()                                
                    t = threading.Thread(target=_f_threadloop, args=(p, q, h, w, e), daemon=True)
                    t.start()

                    frameindex = 0
                    while True:                        
                        img = q.get()
                        if img is not None:
                            im = self._video.frame(frameindex, img)
                            frameindex += 1
                            yield im
                        else:
                            e.set()
                            break
                        
            def __getitem__(self, k):
                """Retrieve individual frame index - this is inefficient, use __iter__ instead"""
                return self._video.preview(frame=k)  # this is inefficient

            def write(self, im, flush=False):
                """Write individual frames to write stream"""
                
                assert isinstance(im, vipy.image.Image)
                if self._shape is None:
                    self._shape = im.shape()
                    assert im.channels() == 3, "RGB frames required"
                    self.__enter__()
                assert self._write_pipe is not None, "Write stream cannot be initialized"                
                assert im.shape() == self._shape, "Shape cannot change during writing"
                self._write_pipe.stdin.write(im.array().astype(np.uint8).tobytes())
                if flush:
                    self._write_pipe.stdin.flush()  # do we need this?
                if isinstance(im, vipy.image.Scene) and len(im.objects()) > 0:
                    for obj in im.objects():
                        self._video.add(obj, frame=self._frame_index, rangecheck=False)
                self._frame_index += 1  # assumes that the source image is at the appropriate frame rate for this video

            def clip(self, n, m=1, continuous=False, tracks=True, activities=True, delay=0):
                """Stream clips of length n such that the yielded video clip contains frame(0+delay) to frame(n+delay), and next contains frame(m+delay) to frame(n+m+delay). 

                Usage examples:
                
                >>> for vc in v.stream().clip(n=16, m=2):
                >>>     # yields video vc with frames [0,16] from v
                >>>     # then video vc with frames [2,18] from v
                >>>     # ... finally video with frames [len(v)-n-1, len(v)-1]

                Introducing a delay so that the clips start at a temporal offset from v

                >>> for vc in v.stream().clip(n=8, m=3, delay=1):
                >>>     # yields video vc with frames [1,9]
                >>>     # then video vc with frames [4,12] ...

                Args:
                    n: [int] the length of the clip in frames
                    m: [int] the stride between clips in frames
                    delay: [int] The temporal delay in frames for the clip, must be less than n and >= 0
                    continuous: [bool]  if true, then yield None for the sequential frames not aligned with a stride so that a clip is yielded on every frame
                    activities: [bool]  if false, then activities from the source video are not copied into the clip
                    tracks: [bool]  if false, then tracks from the source video are not copied into the clip

                Returns:
                    An iterator that yields `vipy.video.Video` objects each of length n with startframe += m, starting at frame=delay, such that each video contains the tracks and activities (if requested) for this clip sourced from the shared stream video.
                """
                assert isinstance(n, int) and n>0, "Clip length must be a positive integer"
                assert isinstance(m, int) and m>0, "Clip stride must be a positive integer"
                assert isinstance(delay, int) and delay >= 0 and delay < n, "Clip delay must be a positive integer less than n"
                
                def _f_threadloop(pipe, queue, height, width, video, n, m, continuous, event):
                    assert self._video.isloaded() or pipe is not None, "invalid pipe"
                    assert queue is not None, "invalid queue"
                    frameindex = 0
                    frames = []
                    
                    while True:
                        if self._video.isloaded():
                            if frameindex < len(self._video):
                                frames.append(self._video[frameindex].array())
                            else:
                                queue.put( (None, None) )
                                event.wait()                                
                                break
                        else:
                            in_bytes = pipe.stdout.read(height * width * 3)
                            if not in_bytes:
                                queue.put( (None, None) )
                                pipe.poll()
                                if pipe.returncode != 0:
                                    #raise ValueError('Clip stream iterator exited')
                                    pass
                                event.wait()
                                break
                            else:
                                frames.append(np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))

                        if len(frames) > n:
                            frames.pop(0)
                        if ((frameindex+1-delay) % m) == 0 and len(frames) >= n:
                            # Use frameindex+1 so that we include (0,1), (1,2), (2,3), ... for n=2, m=1
                            # The delay shifts the clip +delay frames (1,2,3), (3,4,5), ... for n=3, m=2, delay=1
                            queue.put( (frameindex, video.clear().clone(shallow=True).array(np.stack(frames[-n:]))))  # requires copy, expensive operation                            
                        elif continuous:
                            queue.put((frameindex, None))
                            
                        frameindex += 1

                p = self._read_pipe()
                q = queue.Queue(self._queuesize)
                (h, w) = self._shape                        
                v = self._video.clone(flushfilter=True).clear().nourl().nofilename()
                e = threading.Event()                
                t = threading.Thread(target=_f_threadloop, args=(p, q, h, w, v, n, m, continuous, e), daemon=True)
                t.start()

                while True:
                    (k,vc) = q.get()
                    if k is not None:
                        # Yield clip such that activities and tracks are relative to frame 0 if the clip
                        # - self._video is shared and contains the current activities and tracks
                        # - The frameindex (k) is the frame index in self._video that corresponds to the last frame in the clip.
                        # - The clip contains n frames from (k-(n-1), k), recall k is zero indexed, n is length which is one indexed so subtract n-1 
                        # - Offset all activities and tracks by -(k-(n-1)) so that they are relative to frame 0 of clip
                        # - Truncate is inclusive, so use n-1 as the last frame
                        yield ((vc.activities([a.clone().offset(-(k-(n-1))).truncate(0,n-1) for (ak,a) in self._video.activities().items() if a.during_interval(k-(n-1), k, inclusive=False)] if activities else []) 
                                .tracks([t.clone(k-(n-1), k).offset(-(k-(n-1))).truncate(0,n-1) for (tk,t) in self._video.tracks().items() if t.during_interval(k-(n-1), k)] if tracks else []))
                               if (vc is not None and isinstance(vc, vipy.video.Scene)) else vc)
                    else:
                        e.set()
                        break
                    
            def batch(self, n):
                """Stream batches of length n such that each batch contains frames [0,n], [n+1, 2n], ...  Last batch will be ragged.

                .. warning:: Unlike clip(), this method currently does not support activities and tracks.  The batch will contain pixels only. 
                """
                assert isinstance(n, int) and n>0, "batch length must be a positive integer"

                def _f_threadloop(pipe, queue, height, width, video, n, event):
                    assert self._video.isloaded() or pipe is not None, "invalid pipe"
                    assert queue is not None, "invalid queue"
                    frameindex = 0
                    frames = []

                    while True:
                        if self._video.isloaded():
                            if frameindex < len(self._video):
                                frames.append(self._video[frameindex].array())
                            else:
                                if len(frames) > 0:
                                    queue.put((frameindex, video.clear().clone(shallow=True).array(np.stack(frames))))                                
                                queue.put( (None, None) )
                                event.wait()                                
                                break
                        else:                            
                            in_bytes = pipe.stdout.read(height * width * 3)
                            if not in_bytes:
                                if len(frames) > 0:
                                    queue.put((frameindex, video.clear().clone(shallow=True).array(np.stack(frames))))
                                queue.put((None, None))
                                pipe.poll()
                                if pipe.returncode != 0:
                                    #raise ValueError('Batch stream iterator exited')
                                    pass
                                event.wait()
                                break
                            else:
                                frames.append(np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))
                            
                        if len(frames) == n:
                            queue.put( (frameindex, video.clear().clone(shallow=True).array(np.stack(frames))) )  # requires copy, expensive operation                            
                            frames = []
                            
                        frameindex += 1

                p = self._read_pipe()
                q = queue.Queue(self._queuesize)
                (h, w) = self._shape                        
                v = self._video.clone(flushfilter=True).clear().nourl().nofilename()
                e = threading.Event()
                t = threading.Thread(target=_f_threadloop, args=(p, q, h, w, v, n, e), daemon=True)
                t.start()
                
                while True:
                    (k,vb) = q.get()
                    if k is not None:
                        yield vb  # FIXME: should clone shift and truncate activities/tracks
                    else:
                        e.set()
                        break

            def frame(self, n=0):
                """Stream individual frames of video with negative offset n to the stream head. If n=-30, this will return a frame 30 frames ago"""
                assert isinstance(n, int) and n<=0, "Frame offset must be non-positive integer"
                frames = []
                for (k,im) in enumerate(self):
                    frames.append(im)                    
                    imout = frames[0]
                    frames.pop(0) if len(frames) == abs(n) else None
                    yield imout
                                          
        return Stream(self)  # do not clone


    def clear(self):
        """no-op for `vipy.video.Video` object, used only for `vipy.video.Scene`"""
        return self
    
    def bytes(self):
        """Return a bytes representation of the video file"""
        assert self.hasfilename(), "Invalid filename"
        with open(self.filename(), 'rb') as f:
            data = io.BytesIO(f.read())
        return str(data.read()).encode('UTF-8')
    
    def frames(self):
        """Alias for __iter__()"""
        return self.__iter__()
                
    def framelist(self):
        return list(self.frames())

    def _update_ffmpeg_seek(self, timestamp_in_seconds=0, offset=0):
        if timestamp_in_seconds == 0 and offset == 0:
            return self
        nodes = ffmpeg.nodes.get_stream_spec_nodes(self._ffmpeg)
        sorted_nodes, outgoing_edge_maps = ffmpeg.dag.topo_sort(nodes)
        for n in sorted_nodes:
            if 'input' == n.__dict__['name']:
                if 'ss' not in n.__dict__['kwargs']:
                    n.__dict__['kwargs']['ss'] = 0
                if timestamp_in_seconds == 0:
                    n.__dict__['kwargs']['ss'] = n.__dict__['kwargs']['ss'] + offset
                else: 
                    n.__dict__['kwargs']['ss'] = timestamp_in_seconds + offset                   
                return self
        raise ValueError('invalid ffmpeg argument "%s" -> "%s"' % ('ss', timestamp_in_seconds))

        
    def _update_ffmpeg(self, argname, argval, node_name=None):
        """Update the ffmpeg filter chain to overwrite the (argname, argval) elements. 

        Useful for fine-tuning a filter chain without rewwriring the whole thing.
        """
        nodes = ffmpeg.nodes.get_stream_spec_nodes(self._ffmpeg)
        sorted_nodes, outgoing_edge_maps = ffmpeg.dag.topo_sort(nodes)
        for n in sorted_nodes:
            if argname in n.__dict__['kwargs'] or node_name == n.__dict__['name']:
                n.__dict__['kwargs'][argname] = argval
                return self
        raise ValueError('invalid ffmpeg argument "%s" -> "%s"' % (argname, argval))
               
    def _ffmpeg_commandline(self, f=None):
        """Return the ffmpeg command line string that will be used to process the video"""
        cmd = f.compile() if f is not None else self._ffmpeg.output('dummyfile').compile()
        for (k,c) in enumerate(cmd):
            if c is None:
                cmd[k] = str(c)
            elif 'filter' in c:
                cmd[k+1] = '"%s"' % str(cmd[k+1])
            elif 'map' in c:
                cmd[k+1] = '"%s"' % str(cmd[k+1])
        return str(' ').join(cmd)

    def commandline(self):
        """Return the equivalent ffmpeg command line string that will be used to transcode the video.
        
           This is useful for introspecting the complex filter chain that will be used to process the video.  You can try to run this command line yourself for debugging purposes, by replacing 'dummyfile' with an appropriately named output file.
        """        
        return self._ffmpeg_commandline()
    
    def _from_ffmpeg_commandline(self, cmd, strict=False):
        """Convert the ffmpeg command line string (e.g. from `vipy.video.Video.commandline`) to the corresponding ffmpeg-python filter chain and update self"""
        args = copy.copy(cmd).replace(str(self.filename()), 'FILENAME').split(' ')  # filename may contain spaces
        
        assert args[0] == 'ffmpeg', "Invalid FFMEG commmand line '%s'" % cmd
        assert args[1] == '-i' or (args[3] == '-i' and args[1] == '-ss'), "Invalid FFMEG commmand line '%s'" % cmd
        assert args[-1] == 'dummyfile', "Invalid FFMEG commmand line '%s'" % cmd
        assert len(args) >= 4, "Invalid FFMEG commmand line '%s'" % cmd

        if args[1] == '-ss':
            timestamp_in_seconds = float(args[2])
            timestamp_in_seconds = int(timestamp_in_seconds) if timestamp_in_seconds == 0 else timestamp_in_seconds  # 0.0 -> 0
            args = [args[0]] + args[3:]
            f = ffmpeg.input(args[2].replace('FILENAME', self.filename()), ss=timestamp_in_seconds)   # restore filename, set offset time
            self._startframe = int(round(timestamp_in_seconds*self.framerate()))  # necessary for clip() and __repr__
        else:
            f = ffmpeg.input(args[2].replace('FILENAME', self.filename()))  # restore filename

        if len(args) > 4:
            assert args[3] == '-filter_complex', "Invalid FFMEG commmand line '%s'" % cmd
            assert args[4][0] == '"' and args[4][-1] == '"', "Invalid FFMEG commmand line '%s'" % cmd

            filterargs = args[4][1:-1].split(';')
            for a in filterargs:
                assert a.count(']') == 2 and a.count('[') == 2
                kwstr = a.split(']', maxsplit=1)[1].split('[', maxsplit=1)[0]
                if kwstr.count('=') == 0:
                    f = f.filter(kwstr)
                else:
                    (a, kw) = ([], {}) 
                    (filtername, kwstr) = kwstr.split('=', maxsplit=1)
                    for s in kwstr.split(':'):
                        if s.count('=') > 0:
                            (k,v) = s.split('=')
                            kw[k] = v
                        else:
                            a.append(s)

                    if 'end' in kw:
                        self._endframe = (self._startframe if self._startframe is not None else 0) + int(round(float(kw['end'])*self.framerate()))  # for __repr__
                    if 'start' in kw:
                        pass
                    if 'start_frame' in kw or 'end_frame' in kw:
                        f = f.setpts('PTS-STARTPTS')  # reset timestamp to 0 before trim filter in seconds
                        if 'end_frame' in kw:
                            self._endframe = (self._startframe if self._startframe is not None else 0) + int(kw['end_frame'])  # for __repr__
                            kw['end'] = int(kw['end_frame'])/self.framerate()  # convert end_frame to end (legacy)
                            del kw['end_frame']  # use only end and not end frame
                        if 'start_frame' in kw:
                            self._startframe = (self._startframe if self._startframe is not None else 0) + int(kw['start_frame'])  # for __repr__
                            kw['start'] = int(kw['start_frame'])/self.framerate()  # convert start_frame to start (legacy)
                            del kw['start_frame']  # use only start and not start_frame

                    f = f.filter(filtername, *a, **kw)

        if strict:
            assert self._ffmpeg_commandline(f.output('dummyfile')) == cmd, "FFMPEG command line '%s' != '%s'" % (self._ffmpeg_commandline(f.output('dummyfile')), cmd)
        return f

    def _isdirty(self):
        """Has the FFMPEG filter chain been modified from the default?  If so, then ffplay() on the video file will be different from self.load().play()"""
        return '-filter_complex' in self._ffmpeg_commandline()

    def probeshape(self):
        """Return the (height, width) of underlying video file as determined from ffprobe
        
        .. warning:: this does not take into account any applied ffmpeg filters.  The shape will be the (height, width) of the underlying video file.  
        """
        p = self.probe()
        assert len(p['streams']) > 0
        return (p['streams'][0]['height'], p['streams'][0]['width'])
        
    def duration_in_seconds_of_videofile(self):
        """Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe.  Fetch once and cache.
        
        .. notes:: This is the duration of the source video and NOT the duration of the filter chain.  If you load(), this may be different duration depending on clip() or framerate() directives.
        """
        filehash = hashlib.md5(str(self.filename()).encode()).hexdigest()            
        if self.hasattribute('_duration_in_seconds_of_videofile') and self.attributes['_duration_in_seconds_of_videofile']['filehash'] == filehash:
            return self.attributes['_duration_in_seconds_of_videofile']['duration']
        else:
            d = float(self.probe()['format']['duration'])
            self.attributes['_duration_in_seconds_of_videofile'] = {'duration':d, 'filehash':filehash}  # for next time
            return d

    def duration_in_frames_of_videofile(self):
        """Return video duration of the source video file (NOT the filter chain) in frames, requires ffprobe.

        .. notes:: This is the duration of the source video and NOT the duration of the filter chain.  If you load(), this may be different duration depending on clip() or framerate() directives.
        """
        return int(np.floor(self.duration_in_seconds_of_videofile()*self.framerate_of_videofile()))
    
    def framerate_of_videofile(self):
        """Return video framerate in frames per second of the source video file (NOT the filter chain), requires ffprobe.
        """
        p = self.probe()
        assert 'streams' in p and len(['streams']) > 0
        fps = p['streams'][0]['avg_frame_rate']
        return float(fps) if '/' not in fps else (float(fps.split('/')[0]) / float(fps.split('/')[1]))  # fps='30/1' or fps='30.0'

    def probe(self):
        """Run ffprobe on the filename and return the result as a dictionary"""
        if not has_ffprobe:
            raise ValueError('"ffprobe" executable not found on path, this is optional for vipy.video - Install from http://ffmpeg.org/download.html')            
        assert self.hasfilename(), "Invalid video file '%s' for ffprobe" % self.filename() 
        return ffmpeg.probe(self.filename())

    def print(self, prefix='', verbose=True, sleep=None):
        """Print the representation of the video

        This is useful for debugging in long fluent chains.  Sleep is useful for adding in a delay for distributed processing.

        Args:
            prefix: prepend a string prefix to the video __repr__ when printing.  Useful for logging.
            verbose:  Print out the video __repr__.  Set verbose=False to just sleep
            sleep:  Integer number of seconds to sleep[ before returning

        Returns:  
            The video object after sleeping 
        """
        if verbose:
            print(prefix+self.__repr__())
        if sleep is not None:
            assert isinstance(sleep, int) and sleep > 0, "Sleep must be a non-negative integer number of seconds"
            time.sleep(sleep)
        return self

    def __array__(self):
        """Called on np.array(self) for custom array container, (requires numpy >=1.16)"""
        return self.numpy()

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding."""
        return self.json(encode=False)

    def json(self, encode=True):
        """Return a json representation of the video.
        
        Args:
            encode: If true, return a JSON encoded string using json.dumps
        
        Returns:
            A JSON encoded string if encode=True, else returns a dictionary object 

        .. note::  If the video is loaded, then the JSON will not include the pixels.  Try using `vipy.video.Video.store` to serialize videos, or call `vipy.video.Video.flush` first.
        """
        
        if self.isloaded():
            warnings.warn("JSON serialization of video requires flushed buffers, will not include the loaded video.  Try store()/restore()/unstore() instead to serialize videos as standalone objects efficiently.")
        d = {'_filename':self._filename,
             '_url':self._url,
             '_framerate':self._framerate,
             '_array':None,
             '_colorspace':self._colorspace,
             'attributes':self.attributes,
             '_startframe':self._startframe,
             '_endframe':self._endframe,
             '_endsec':self._endsec,
             '_startsec':self._startsec,
             '_ffmpeg':self._ffmpeg_commandline()}
        return json.dumps(d) if encode else d
    
    def take(self, n):
        """Return n frames from the clip uniformly spaced as numpy array
        
        Args:
            n: Integer number of uniformly spaced frames to return 
        
        Returns:
            A numpy array of shape (n,W,H)

        .. warning:: This assumes that the entire video is loaded into memory (e.g. call `vipy.video.Video.load`).  Use with caution.
        """
        assert self.isloaded(), "load() is required before take()"
        dt = int(np.round(len(self._array) / float(n)))  # stride
        return self._array[::dt][0:n]

    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations

        Args:
            fps: Float frames per second to process the underlying video

        Returns:
            If fps is None, return the current framerate, otherwise set the framerate to fps

        """
        if fps is None:
            return self._framerate
        elif fps == self._framerate:
            return self
        else:            
            assert not self.isloaded(), "Filters can only be applied prior to load()"
            if 'fps=' in self._ffmpeg_commandline():
                self._update_ffmpeg('fps', fps)  # replace fps filter, do not add to it
            else:
                self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')  # create fps filter first time
        
            # if '-ss' in self._ffmpeg_commandline():
            #     No change is needed here.  The seek is in seconds and is independent of the framerate
            # if 'trim' in self._ffmpeg_commandline():
            #     No change is needed here.  The trim is in units of seconds which is independent of the framerate

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
        """Remove the `vipy.video.Video.url` from the video"""
        (self._url, self._urluser, self._urlpassword, self._urlsha1) = (None, None, None, None)
        return self

    def url(self, url=None, username=None, password=None, sha1=None):
        """Video URL and URL download properties"""
        if url is not None:
            self._url = url  # note that this does not change anything else, better to use the constructor for this
        if url is not None and (isRTSPurl(url) or isRTMPurl(url)):
            self.filename(self._url) 
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

    def isloadable(self, flush=True):
        """Return True if the video can be loaded successfully.
        
        This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.
        
        Args:
            flush: [bool] If true, flush the video after it loads.  This will clear the video pixel buffer

        Returns:
            True if load() can be called without FFMPEG exception.  
            If flush=False, then self will contain the loaded video, which is helpful to avoid load() twice in some conditions
        
        .. warning:: This requires loading and flushing the video.  This is an expensive operation when performed on many videos and may result in out of memory conditions with long videos.  Use with caution!  Try `vipy.video.Video.canload` to test if a single frame can be loaded as a less expensive alternative.
        """
        if not self.isloaded():
            try:
                self.load()  # try to load the whole thing
                if flush:
                    self.flush()
                return True
            except:
                return False
        else:
            return True
        
        
    def canload(self, frame=0):
        """Return True if the video can be previewed at frame=k successfully.
        
        This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.

        .. notes:: This will only try to preview a single frame.  This will not check if the entire video is loadable.  Use `vipy.video.Video.isloadable` in this case
        """
        if not self.isloaded():
            try:
                self.preview(framenum=frame)  # try to preview
                return True
            except:
                return False
        else:
            return True

    def iscolor(self):
        """Is the video a three channel color video as returned from `vipy.video.Video.channels`?"""
        return self.channels() == 3

    def isgrayscale(self):
        """Is the video a single channel as returned from `vipy.video.Video.channels`?"""
        return self.channels() == 1

    def hasfilename(self):
        """Does the filename returned from `vipy.video.Video.filename` exist?"""
        return self._filename is not None and (os.path.exists(self._filename) or isRTSPurl(self._filename) or isRTMPurl(self._filename))

    def isdownloaded(self):
        """Does the filename returned from `vipy.video.Video.filename` exist, meaning that the url has been downloaded to a local file?"""
        return self._filename is not None and os.path.exists(self._filename)

    def hasurl(self):
        """Is the url returned from `vipy.video.Video.url` a well formed url?"""
        return self._url is not None and isurl(self._url)

    def array(self, array=None, copy=False):
        """Set or return the video buffer as a numpy array.
        
        Args:
            array: [np.array] A numpy array of size NxHxWxC = (frames, height, width, channels)  of type uint8 or float32.
            copy: [bool] If true, copy the buffer by value instaed of by reference.  Copied buffers do not share pixels.

        Returns:
            if array=None, return a reference to the pixel buffer as a numpy array, otherwise return the video object.

        """
        if array is None:
            return self._array
        elif isnumpy(array):
            assert array.dtype == np.float32 or array.dtype == np.uint8, "Invalid input - array() must be type uint8 or float32"
            assert array.ndim == 4, "Invalid input array() must be of shape NxHxWxC, for N frames, of size HxW with C channels"
            self._array = np.copy(array) if copy else array
            if copy:
                self._array.setflags(write=True)  # mutable iterators, triggers copy
            self.colorspace(None)  # must be set with colorspace() after array() before _convert()
            return self
        else:
            raise ValueError('Invalid input - array() must be numpy array')            

    def fromarray(self, array):
        """Alias for self.array(..., copy=True), which forces the new array to be a copy"""
        return self.array(array, copy=True)

    def fromdirectory(self, indir, sortkey=None):
        """Create a video from a directory of frames stored as individual image filenames.
        
        Given a directory with files:
        
        framedir/image_0001.jpg
        framedir/image_0002.jpg
        
        >>> vipy.video.Video(frames='/path/to/framedir')
        """
        return self.fromframes([vipy.image.Image(filename=f) for f in sorted(vipy.util.imlist(indir), key=sortkey)])
                                
    def fromframes(self, framelist, copy=True):
        """Create a video from a list of frames"""
        assert all([isinstance(im, vipy.image.Image) for im in framelist]), "Invalid input"
        return self.array(np.stack([im.load().array() if im.load().array().ndim == 3 else np.expand_dims(im.load().array(), 2) for im in framelist]), copy=copy).colorspace(framelist[0].colorspace())
    
    def tonumpy(self):
        """Alias for numpy()"""
        return self.numpy()

    def numpy(self):
        """Convert the video to a writeable numpy array, triggers a load() and copy() as needed"""
        self.load()
        self._array = np.copy(self._array) if not self._array.flags['WRITEABLE'] else self._array  # triggers copy
        self._array.setflags(write=True)  # mutable iterators, torch conversion
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

    def filename(self, newfile=None, copy=False, symlink=False):
        """Update video Filename with optional copy from existing file to new file"""
        if newfile is None:
            return self._filename
        newfile = os.path.normpath(os.path.expanduser(newfile))

        # Copy or symlink from the old filename to the new filename (if requested)
        if copy:
            assert self.hasfilename(), "File not found for copy"
            remkdir(filepath(newfile))
            shutil.copyfile(self._filename, newfile)
        elif symlink:
            assert self.hasfilename(), "File not found for symlink"
            remkdir(filepath(newfile))
            os.symlink(self._filename, newfile)
                    
        # Update ffmpeg filter chain with new input node filename (this file may not exist yet)
        self._update_ffmpeg('filename', newfile)
        self._filename = newfile
        
        return self

    def abspath(self):
        """Change the path of the filename from a relative path to an absolute path (not relocatable)"""
        return self.filename(os.path.normpath(os.path.abspath(os.path.expanduser(self.filename()))))

    def relpath(self, parent=None):
        """Replace the filename with a relative path to parent (or current working directory if none)"""
        parent = parent if parent is not None else os.getcwd()
        assert parent in os.path.expanduser(self.filename()), "Parent path '%s' not found in abspath '%s'" % (parent, self.filename())
        return self.filename(PurePath(os.path.expanduser(self.filename())).relative_to(parent))

    def rename(self, newname):
        """Move the underlying video file preserving the absolute path, such that self.filename() == '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file"""
        newfile = os.path.join(filepath(self.filename()), newname)
        shutil.move(self.filename(), newfile)        
        return self.filename(newfile)
    
    def filesize(self):
        """Return the size in bytes of the filename(), None if the filename() is invalid"""
        return os.path.getsize(self.filename()) if self.hasfilename() else None

    def downloadif(self, ignoreErrors=False, timeout=10, verbose=True, max_filesize='350m'):
        """Download URL to filename if the filename has not already been downloaded"""
        return self.download(ignoreErrors=ignoreErrors, timeout=timeout, verbose=verbose, max_filesize=max_filesize) if self.hasurl() and not self.isdownloaded() else self
    
    def download(self, ignoreErrors=False, timeout=10, verbose=True, max_filesize='350m'):
        """Download URL to filename provided by constructor, or to temp filename.
        
        Args:
            ignoreErrors: [bool] If true, show a warning and return the video object, otherwise throw an exception
            timeout: [int] An integer timeout in seconds for the download to connect
            verbose: [bool] If trye, show more verbose console output
            max_filesize: [str] A string of the form 'NNNg' or 'NNNm' for youtube downloads to limit the maximum size of a URL to '350m' 350MB or '12g' for 12GB.

        Returns:
            This video object with the video downloaded to the filename()        
        """
        if self._url is None and self._filename is not None:
            return self
        if self._url is None:
            raise ValueError('[vipy.video.download]: No URL to download')
        elif not isurl(str(self._url)):
            raise ValueError('[vipy.video.download]: Invalid URL "%s" ' % self._url)

        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if isyoutubeurl(self._url):
                f = self._filename if filefull(self._filename) is None else filefull(self._filename)
                vipy.videosearch.download(self._url, f, writeurlfile=False, skip=ignoreErrors, verbose=verbose, max_filesize=max_filesize)
                for ext in ['mkv', 'mp4', 'webm']:
                    f = '%s.%s' % (self.filename(), ext)
                    if os.path.exists(f):
                        self.filename(f)  # change the filename to match the youtube extension
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
                    if vipy.globals.cache() is not None:
                        self.filename(os.path.join(remkdir(vipy.globals.cache()), filetail(self._url)))
                vipy.downloader.s3(self.url(), self.filename(), verbose=verbose)
                    
            elif url_scheme == 'scp':                
                if self.filename() is None:
                    self.filename(templike(self._url))                    
                    if vipy.globals.cache() is not None:
                        self.filename(os.path.join(remkdir(vipy.globals.cache()), filetail(self._url)))
                vipy.downloader.scp(self._url, self.filename(), verbose=verbose)
 
            elif not isvideourl(self._url) and vipy.videosearch.is_downloadable_url(self._url):
                vipy.videosearch.download(self._url, filefull(self._filename), writeurlfile=False, skip=ignoreErrors, verbose=verbose, max_filesize=max_filesize)
                for ext in ['mkv', 'mp4', 'webm']:
                    f = '%s.%s' % (self.filename(), ext)
                    if os.path.exists(f):
                        self.filename(f)
                        break    
                if not self.hasfilename():
                    raise ValueError('Downloaded filenot found "%s.*"' % self.filename())

            elif url_scheme == 'rtsp':
                # https://ffmpeg.org/ffmpeg-protocols.html#rtsp
                pass
            
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

    def shape(self, shape=None, probe=False):
        """Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user"""
        if probe:
            return self.shape(self.probeshape(), probe=False)
        elif shape is not None:
            assert isinstance(shape, tuple), "shape=(height, width) tuple"
            self._shape = shape
            self._channels = self.channels()
            #self._previewhash = hashlib.md5(str(self._ffmpeg_commandline()).encode()).hexdigest() 
            return self
            
        elif not self.isloaded():
            #previewhash = hashlib.md5(str(self._ffmpeg_commandline()).encode()).hexdigest()
            #if not hasattr(self, '_previewhash') or previewhash != self._previewhash:
            if self._shape is None or len(self._shape) == 0:  # dirty filter chain
                im = self.preview()  # ffmpeg chain changed, load a single frame of video, triggers fetch
                self._shape = (im.height(), im.width())  # cache the shape
                self._channels = im.channels()
                #self._previewhash = previewhash
            return self._shape
        else:
            return (self._array.shape[1], self._array.shape[2])

    def channels(self):
        """Return integer number of color channels"""
        if not self.isloaded():
            self._channels = 3   # always color video 
            #previewhash = hashlib.md5(str(self._ffmpeg_commandline()).encode()).hexdigest()            
            #if not hasattr(self, '_previewhash') or previewhash != self._previewhash:
            #    im = self.preview()  # ffmpeg chain changed, load a single frame of video
            #    self._shape = (im.height(), im.width())  # cache the shape                
            #    self._channels = im.channels()  # cache
            #    self._previewhash = previewhash
            return self._channels  # cached
        else:
            return 1 if self.load().array().ndim == 3 else self.load().array().shape[3]
        
    def width(self):
        """Width (cols) in pixels of the video for the current filter chain"""
        return self.shape()[1]

    def height(self):
        """Height (rows) in pixels of the video for the current filter chain"""
        return self.shape()[0]

    def aspect_ratio(self):
        """The width/height of the video expressed as a fraction"""
        return self.width() / self.height()

    def preview(self, framenum=0):
        """Return selected frame of filtered video, return vipy.image.Image object.  This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video"""
        if self.isloaded():
            return self[framenum]
        elif self.hasurl() and not self.hasfilename():
            self.download(verbose=True)  
        if not self.hasfilename():
            raise ValueError('Video file not found')

        # Convert frame to mjpeg and pipe to stdout, used to get dimensions of video
        #   - The MJPEG encoder will generally output lower quality than H.264 encoded frames
        #   - This means that frame indexing from preview() will generate slightly different images than streaming raw
        #   - Beware running convnets, as the pixels will be slightly different (~4 grey levels in uint8) ... 
        try:
            # FFMPEG frame indexing is inefficient for large framenum.  Need to add "-ss sec.msec" flag before input
            #   - the "ss" option must be provided before the input filename, and is supported by ffmpeg-python as ".input(in_filename, ss=time)"
            #   - Seek to the frame before the desired frame in order to pipe the next (desired) frame 
            timestamp_in_seconds = max(0.0, (framenum-1)/float(self.framerate()))
            f_prepipe = self.clone(shallow=True)._update_ffmpeg_seek(offset=timestamp_in_seconds)._ffmpeg.filter('select', 'gte(n,{})'.format(0))
            f = f_prepipe.output('pipe:', vframes=1, format='image2', vcodec='mjpeg')\
                         .global_args('-cpuflags', '0', '-loglevel', 'debug' if vipy.globals.isdebug() else 'error')
            (out, err) = f.run(capture_stdout=True, capture_stderr=True)
        except Exception as e:            
            raise ValueError('[vipy.video.load]: Video preview failed with error "%s"\n  - Video: "%s"\n  - FFMPEG command: \'sh> %s\'\n  - Try manually running this ffmpeg command to see errors.  This error usually means that the video is corrupted.' % (str(e), str(self), str(self._ffmpeg_commandline(f_prepipe.output('preview.jpg', vframes=1)))))

        # [EXCEPTION]:  UnidentifiedImageError: cannot identify image file, means usually that FFMPEG piped a zero length image
        try:
            return Image(array=np.array(PIL.Image.open(BytesIO(out))))
        except Exception as e:
            print('[vipy.video.Video.preview][ERROR]:  %s' % str(e))
            print('  - FFMPEG attempted to extract a single frame from the following video and failed:\n    %s' % str(self))
            print('  - This may occur after calling clip() with too short a duration, try increasing the clip to be > 1 sec')
            print('  - This may occur after calling clip() with a startframe or endframe outside the duration of the video')
            print('  - This may occur if requesting a frame number greater than the length of the video.  At this point, we do not know the video length, and cannot fail gracefully')
            print('  - This may occur when the framerate of the video from ffprobe (tbr) does not match that passed to fps filter, resulting in a zero length image preview piped to stdout')
            print('  - This may occur if the filter chain fails for some unknown reason on this video.  Try running this ffmpeg command manually and inspect the FFMPEG console output:\n     sh> %s' % str(self._ffmpeg_commandline(f_prepipe.output('preview.jpg', vframes=1))))
            raise

    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile.

        This is functionally equivalent to `vipy.video.Video.frame` with an additional outfile argument to easily save an annotated thumbnail image.

        Args:
            outfile: [str] an optional outfile to save the annotated frame 
            frame: [int >= 0] The frame to output the thumbnail

        Returns:
            A `vipy.image.Image` object for frame k.  
        """
        im = self.frame(frame, img=self.preview(frame).array())
        return im.savefig(outfile) if outfile is not None else im
    
    def load(self, verbose=False, ignoreErrors=False, shape=None):
        """Load a video using ffmpeg, applying the requested filter chain.  
           
        Args:
            verbose: [bool] if True. then ffmpeg console output will be displayed. 
            ignoreErrors: [bool] if True, then all load errors are warned and skipped.  Be sure to call isloaded() to confirm loading was successful.
            shape: [tuple (height, width, channels)]  If provided, use this shape for reading and reshaping the byte stream from ffmpeg.  This is useful for efficient loading in some scenarios. Knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size

        Returns:
            this video object, with the pixels loaded in self.array()

        .. warning:: Loading long videos can result in out of memory conditions.  Try to call clip() first to extract a video segment to load().
        """
        if self.isloaded():
            return self
        elif not self.hasfilename() and self.hasurl():
            self.download(ignoreErrors=ignoreErrors)
        elif not self.hasfilename() and not ignoreErrors:
            raise ValueError('Invalid input - load() requires a valid URL, filename or array')
        if not self.hasfilename() and ignoreErrors:
            print('[vipy.video.load]: Video file "%s" not found - Ignoring' % self.filename())
            return self
        if verbose:
            print('[vipy.video.load]: Loading "%s"' % self.filename())

        # Load the video
        # 
        # [EXCEPTION]:  older ffmpeg versions may segfault on complex crop filter chains
        #    -On some versions of ffmpeg setting -cpuflags=0 fixes it, but the right solution is to rebuild from the head (30APR20)
        try:
            f_prepipe = copy.deepcopy(self._ffmpeg)
            f = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24')\
                            .global_args('-cpuflags', '0', '-loglevel', 'debug' if vipy.globals.isdebug() else 'quiet')
            (out, err) = f.run(capture_stdout=True, capture_stderr=True)
        except Exception as e:
            if not ignoreErrors:
                raise ValueError('[vipy.video.load]: Load failed with error "%s"\n\n  - Video: "%s"\n  - FFMPEG command: \'sh> %s\'\n  - This error usually means that the video is corrupted or that you need to upgrade your FFMPEG distribution to the latest stable version.\n  - Try running the output of the ffmpeg command for debugging.' % (str(e), str(self), str(self._ffmpeg_commandline(f_prepipe.output('preview.mp4')))))
            else:
                return self  # Failed, return immediately, useful for calling canload() 

        # Video shape:
        #   - due to complex filter chains, we may not know the final video size without executing it
        #   - However, this introduces extra cost by calling preview() on each filter chain
        #   - If we know what the shape will be (e.g. we made the video square with a known size), then use it here directly
        (height, width, channels) = (self.height(), self.width(), self.channels()) if shape is None else shape
        
        # [EXCEPTION]:  older ffmpeg versions may be off by one on the size returned from self.preview() which uses an image decoder vs. f.run() which uses a video decoder
        #    -Try to correct this manually by searching for a off-by-one-pixel decoding that works.  The right way is to upgrade your FFMPEG version to the FFMPEG head (11JUN20)
        #    -We cannot tell which is the one that the end-user wanted, so we leave it up to the calling function to check dimensions (see self.resize())
        if (len(out) % (height*width*channels)) != 0:
            #warnings.warn('Your FFMPEG version is triggering a known bug that is being worked around in an inefficient manner.  Consider upgrading your FFMPEG distribution.')
            if (len(out) % ((height-1)*(width-1)*channels) == 0):
                (newwidth, newheight) = (width-1, height-1)
            elif (len(out) % ((height-1)*(width)*channels) == 0):
                (newwidth, newheight) = (width, height-1)
            elif (len(out) % ((height-1)*(width+1)*channels) == 0):
                (newwidth, newheight) = (width+1, height-1)
            elif (len(out) % ((height)*(width-1)*channels) == 0):
                (newwidth, newheight) = (width-1, height)
            elif (len(out) % ((height)*(width+1)*channels) == 0):
                (newwidth, newheight) = (width+1, height)
            elif (len(out) % ((height+1)*(width-1)*channels) == 0):
                (newwidth, newheight) = (width-1, height+1)
            elif (len(out) % ((height+1)*(width)*channels) == 0):
                (newwidth, newheight) = (width, height+1)
            elif (len(out) % ((height+1)*(width+1)*channels) == 0):
                (newwidth, newheight) = (width+1, height+1)
            else:
                (newwidth, newheight) = (width, height)

            is_loadable = (len(out) % (newheight*newwidth*channels)) == 0

            if not is_loadable:
                im = self.preview()  # get the real shape...
                (newheight, newwidth, newchannels) = (im.height(), im.width(), im.channels()) 
                        
            assert is_loadable or ignoreErrors, "Load failed for video '%s', and FFMPEG command line: '%s'" % (str(self), str(self._ffmpeg_commandline(f)))
            self._array = np.frombuffer(out, np.uint8).reshape([-1, newheight, newwidth, channels]) if is_loadable else None  # read-only                        
            self.colorspace('rgb' if channels == 3 else 'lum')
            self.resize(rows=height, cols=width)  # Very expensive framewise resizing so that the loaded video is identical shape to preview
        else:
            self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, channels])  # read-only            
            self.colorspace('rgb' if channels == 3 else 'lum')
        return self

    def speed(self, s):
        """Change the speed by a multiplier s.  If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)"""
        assert s > 0, "Invalid input"
        self._ffmpeg = self._ffmpeg.setpts('%1.3f*PTS' % float(1.0/float(s)))
        return self
        
    def clip(self, startframe, endframe=None):
        """Load a video clip betweeen start and end frames"""
        assert (endframe is None or startframe <= endframe) and startframe >= 0, "Invalid start and end frames (%s, %s)" % (str(startframe), str(endframe))
        if not self.isloaded():
            timestamp_in_seconds = ((self._startframe if self._startframe is not None else 0)+startframe)/float(self.framerate())            
            self._update_ffmpeg_seek(timestamp_in_seconds)
            if endframe is not None:
                self._ffmpeg = self._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 before trim filter            
                self._ffmpeg = self._ffmpeg.trim(start=0, end=(endframe-startframe)/self.framerate())  # must be in seconds to allow for framerate conversion
            self._ffmpeg = self._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter            
            self._startframe = startframe if self._startframe is None else self._startframe + startframe  # for __repr__ only
            self._endframe = (self._startframe + (endframe-startframe)) if endframe is not None else endframe  # for __repr__ only
        else:
            endframe = endframe if endframe is not None else len(self._array)
            self._array = self._array[startframe:endframe]
            (self._startframe, self._endframe) = (0, endframe-startframe)
        return self

    def cliprange(self):
        """Return the planned clip (startframe, endframe) range.
        
        This is useful for introspection of the planned clip() before load(), such as for data augmentation purposes without triggering a load. 
        
        Returns:
            (startframe, endframe) of the video() such that after load(), the pixel buffer will contain frame=0 equivalent to startframe in the source video, and frame=endframe-startframe-1 equivalent to endframe in the source video.
            (0, None) If a video does not have a clip() (e.g. clip() was never called, the filter chain does not include a 'trim')

        .. notes:: The endframe can be retrieved (inefficiently) using:

        >>> int(round(self.duration_in_frames_of_videofile() * (self.framerate() / self.framerate_of_videofile())))

        """
        return (self._startframe if self._startframe is not None else 0, self._endframe)

    #def cliptime(self, startsec, endsec):
    #    """Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly"""
    #    assert startsec <= endsec and startsec >= 0, "Invalid start and end seconds (%s, %s)" % (str(startsec), str(endsec))
    #    assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
    #    self._ffmpeg = self._ffmpeg.trim(start=startsec, end=endsec)\
    #                               .setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter
    #    self._startsec = startsec if self._startsec is None else self._startsec + startsec  # for __repr__ only
    #    self._endsec = endsec if self._endsec is None else self._startsec + (endsec-startsec)  # for __repr__ only
    #    return self
    
    def rot90cw(self):
        """Rotate the video 90 degrees clockwise, can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self.shape(shape=(self.width(), self.height()))  # transposed        
        self._ffmpeg = self._ffmpeg.filter('transpose', 1)
        return self

    def rot90ccw(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self.shape(shape=(self.width(), self.height()))  # transposed                
        self._ffmpeg = self._ffmpeg.filter('transpose', 2)
        return self

    def fliplr(self):
        """Mirror the video left/right by flipping horizontally"""
        if not self.isloaded():
            self._ffmpeg = self._ffmpeg.filter('hflip')
        else:
            self.array(np.stack([np.fliplr(f) for f in self._array]), copy=False)
        return self

    def flipud(self):
        """Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()"""        
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self._ffmpeg = self._ffmpeg.filter('vflip')
        return self

    def rescale(self, s):
        """Rescale the video by factor s, such that the new dimensions are (s*H, s*W), can only be applied prior to load()"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        self.shape(shape=(int(np.round(self.height()*float(np.ceil(s*1e6)/1e6))), int(np.round(self.width()*float(np.ceil(s*1e6)/1e6)))))  # update the known shape        
        self._ffmpeg = self._ffmpeg.filter('scale', 'iw*%1.6f' % float(np.ceil(s*1e6)/1e6), 'ih*%1.6f' % float(np.ceil(s*1e6)/1e6))  # ceil last significant digit to avoid off by one
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to be (rows=height, cols=width)"""
        newshape = (rows if rows is not None else int(np.round(self.height()*(cols/self.width()))),
                    cols if cols is not None else int(np.round(self.width()*(rows/self.height()))))
                            
        if (rows is None and cols is None):
            return self  # only if strictly necessary
        if not self.isloaded():
            self._ffmpeg = self._ffmpeg.filter('scale', cols if cols is not None else -1, rows if rows is not None else -1)
        else:            
            # Do not use self.__iter__() which triggers copy for mutable arrays
            #self.array(np.stack([Image(array=self._array[k]).resize(rows=rows, cols=cols).array() for k in range(len(self))]), copy=False)
            
            # Faster: RGB->RGBX to allow for PIL.Image.fromarray() without tobytes() copy, padding faster than np.concatenate()
            #self.array(np.stack([PIL.Image.fromarray(x, mode='RGBX').resize( (cols, rows), resample=PIL.Image.BILINEAR) for x in np.pad(self._array, ((0,0),(0,0),(0,0),(0,1)))])[:,:,:,:-1], copy=False)  # RGB->RGBX->RGB
            
            # Fastest: padding introduces more overhead than just accepting tobytes(), image size dependent?
            self.array(np.stack([PIL.Image.fromarray(x).resize( (cols, rows), resample=PIL.Image.BILINEAR) for x in np.ascontiguousarray(self._array)]), copy=False)
        self.shape(shape=newshape)  # manually set newshape
        return self

    def mindim(self, dim=None):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert dim is None or not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return min(self.shape()) if dim is None else (self.resize(cols=dim) if W<H else self.resize(rows=dim))

    def maxdim(self, dim=None):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return max(H,W) if dim is None else (self.resize(cols=dim) if W>H else self.resize(rows=dim))
    
    def randomcrop(self, shape, withbox=False):
        """Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box"""
        assert shape[0] <= self.height() and shape[1] <= self.width()  # triggers preview()
        (xmin, ymin) = (np.random.randint(self.height()-shape[0]), np.random.randint(self.width()-shape[1]))
        bb = vipy.geometry.BoundingBox(xmin=int(xmin), ymin=int(ymin), width=int(shape[1]), height=int(shape[0]))  # may be outside frame
        self.crop(bb, zeropad=True)
        return self if not withbox else (self, bb)

    def centercrop(self, shape, withbox=False):
        """Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box"""
        assert shape[0] <= self.height() and shape[1] <= self.width()  # triggers preview()
        bb = vipy.geometry.BoundingBox(xcentroid=float(self.width()/2.0), ycentroid=float(self.height()/2.0), width=float(shape[1]), height=float(shape[0])).int()  # may be outside frame
        self.crop(bb, zeropad=True)
        return self if not withbox else (self, bb)

    def centersquare(self):
        """Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant"""
        return self.centercrop( (min(self.height(), self.width()), min(self.height(), self.width())))

    def cropeven(self):
        """Crop the video to the largest even (width,height) less than or equal to current (width,height).  This is useful for some codecs or filters which require even shape."""
        return self.crop(vipy.geometry.BoundingBox(xmin=0, ymin=0, width=int(vipy.math.even(self.width())), height=int(vipy.math.even(self.height()))))
    
    def maxsquare(self):
        """Pad the video to be square, preserving the upper left corner of the video"""
        # This ffmpeg filter can throw the error:  "Padded dimensions cannot be smaller than input dimensions." since the preview is off by one.  Add one here to make sure.
        # FIXME: not sure where in some filter chains this off-by-one error is being introduced, but probably does not matter since it does not affect any annotations 
        # and since the max square always preserves the scale and the upper left corner of the source video. 
        # FIXME: this may trigger an inefficient resizing operation during load()
        d = max(self.shape())
        self._ffmpeg = self._ffmpeg.filter('pad', d+1, d+1, 0, 0)
        self.shape(shape=(d+1, d+1))
        return self.crop(vipy.geometry.BoundingBox(xmin=0, ymin=0, width=int(d), height=int(d)))

    def minsquare(self):
        """Return a square crop of the video, preserving the upper left corner of the video"""
        d = min(self.shape())
        self.shape(shape=(d, d))
        return self.crop(vipy.geometry.BoundingBox(xmin=0, ymin=0, width=int(d), height=int(d)))
    
    def maxmatte(self):
        return self.zeropad(max(1, int((max(self.shape()) - self.width())/2)), max(int((max(self.shape()) - self.height())/2), 1))
    
    def zeropad(self, padwidth, padheight):
        """Zero pad the video with padwidth columns before and after, and padheight rows before and after
           
        .. notes:: Older FFMPEG implementations can throw the error "Input area #:#:#:# not within the padded area #:#:#:# or zero-sized, this is often caused by odd sized padding. 
             Recommend calling self.cropeven().zeropad(...) to avoid this

        """
        assert isinstance(padwidth, int) and isinstance(padheight, int)        
        if not self.isloaded():
            self.shape(shape=(self.height()+2*padheight, self.width()+2*padwidth))  # manually set shape to avoid preview            
            self._ffmpeg = self._ffmpeg.filter('pad', 'iw+%d' % (2*padwidth), 'ih+%d' % (2*padheight), '%d'%padwidth, '%d'%padheight)
        elif padwidth > 0 or padheight > 0:
            self.array( np.pad(self.array(), ((0,0), (padheight,padheight), (padwidth,padwidth), (0,0)), mode='constant'), copy=False)  # this is very expensive, since np.pad() must copy (once in np.pad >=1.17)            
        return self

    def pad(self, padwidth=0, padheight=0):
        """Alias for zeropad"""
        return self.zeropad(padwidth=padwidth, padheight=padheight)
    
    def crop(self, bbi, zeropad=True):
        """Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().
        """
        assert isinstance(bbi, vipy.geometry.BoundingBox), "Invalid input"
        bbc = bbi.clone().imclipshape(self.width(), self.height()).int()  # clipped box to image rectangle
        bb = bbi.int() if zeropad else bbc  # use clipped box if not zeropad 

        if bb.isdegenerate():
            return None
        elif not self.isloaded():
            if zeropad and bb != bbc:
                # Crop outside the image rectangle will segfault ffmpeg, pad video first (if zeropad=False, then rangecheck will not occur!)
                self.zeropad(int(np.ceil(bb.width()-bbc.width())), int(np.ceil(bb.height()-bbc.height())))     # cannot be called in derived classes
                bb = bb.offset(int(np.ceil(bb.width()-bbc.width())), int(np.ceil(bb.height()-bbc.height())))   # Shift boundingbox by padding (integer coordinates)
            self._ffmpeg = self._ffmpeg.filter('crop', '%d' % bb.width(), '%d' % bb.height(), '%d' % bb.xmin(), '%d' % bb.ymin(), keep_aspect=0)  # keep_aspect=False (disable exact=True, this is not present in older ffmpeg)
        else:
            self.array( bbc.crop(self.array()), copy=False )  # crop first, in-place, valid pixels only
            if zeropad and bb != bbc:
                ((dyb, dya), (dxb, dxa)) = ((max(0, int(abs(np.ceil(bb.ymin() - bbc.ymin())))), max(0, int(abs(np.ceil(bb.ymax() - bbc.ymax()))))),
                                            (max(0, int(abs(np.ceil(bb.xmin() - bbc.xmin())))), max(0, int(abs(np.ceil(bb.xmax() - bbc.xmax()))))))
                self._array = np.pad(self.load().array(), ((0,0), (dyb, dya), (dxb, dxa), (0, 0)), mode='constant')
        self.shape(shape=(bb.height(), bb.width()))  # manually set shape
        return self

    def pkl(self, pklfile=None):
        """save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains"""
        pklfile = pklfile if pklfile is not None else toextension(self.filename(), '.pkl')
        remkdir(filepath(pklfile))
        vipy.util.save(self, pklfile)
        return self

    def pklif(self, b, pklfile=None):
        """Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains"""
        assert isinstance(b, bool)
        return self.pkl(pklfile) if b else self

    def webp(self, outfile, pause=3, strict=True, smallest=False, smaller=False):
        """Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops.  
        
        Args:
            strict: If true, assert that the filename must have an .webp extension
            pause: Integer seconds to pause between loops of the animation
            smallest:  if true, create the smallest possible file but takes much longer to run
            smaller:  If true, create a smaller file, which takes a little longer to run 

        Returns:
            The filename of the webp file for this video

        .. warning::  This may be slow for very long or large videos
        """
        assert strict is False or iswebp(outfile)
        outfile = os.path.normpath(os.path.abspath(os.path.expanduser(outfile)))
        self.load().frame(0).pil().save(outfile, loop=0, save_all=True, method=6 if smallest else 3 if smaller else 0,
                                        append_images=[self.frame(k).pil() for k in range(1, len(self))],
                                        duration=[int(1000.0/self._framerate) for k in range(0, len(self)-1)] + [pause*1000])
        return outfile

    def gif(self, outfile, pause=3, smallest=False, smaller=False):
        """Save a video to an animated GIF file, with pause=N seconds between loops.  

        Args:
            pause: Integer seconds to pause between loops of the animation
            smallest:  If true, create the smallest possible file but takes much longer to run
            smaller:  if trye, create a smaller file, which takes a little longer to run 

        Returns:
            The filename of the animated GIF of this video

        .. warning::  This will be very large for big videos, consider using `vipy.video.Video.webp` instead.
        """        
        assert isgif(outfile)
        return self.webp(outfile, pause, strict=False, smallest=smallest, smaller=True)
        
    def saveas(self, outfile=None, framerate=None, vcodec='libx264', verbose=False, ignoreErrors=False, flush=False, pause=5):
        """Save video to new output video file.  This function does not draw boxes, it saves pixels to a new video file.

        Args:
            outfile: the absolute path to the output video file.  This extension can be .mp4 (for video) or [".webp",".gif"]  (for animated image)
            ignoreErrors:  if True, then exit gracefully without throwing an exception.  Useful for chaining download().saveas() on parallel dataset downloads
            flush:  If true, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel
            framerate:  input framerate of the frames in the buffer, or the output framerate of the transcoded video.  If not provided, use framerate of source video
            pause:  an integer in seconds to pause between loops of animated images if the outfile is webp or animated gif

        Returns:
            a new video object with this video filename, and a clean video filter chain

        .. note:: 
            - If self.array() is loaded, then export the contents of self._array to the video file
            - If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video
            - If outfile==None or outfile==self.filename(), then overwrite the current filename 

        """        
        outfile = tocache(tempMP4()) if outfile is None else os.path.normpath(os.path.abspath(os.path.expanduser(outfile)))
        premkdir(outfile)  # create output directory for this file if not exists
        framerate = framerate if framerate is not None else self._framerate

        if verbose:
            print('[vipy.video.saveas]: Saving video "%s" ...' % outfile)                      
        try:
            if iswebp(outfile):
                return self.webp(outfile, pause)
            elif isgif(outfile):
                return self.gif(outfile, pause)
            elif isjsonfile(outfile):
                with open(outfile) as f:
                    f.write(self.json(encode=True))
                return outfile
            elif self.isloaded():
                # Save numpy() from load() to video, forcing to be even shape
                (n, height, width, channels) = self._array.shape
                process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate) \
                                .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                                .output(filename=outfile, pix_fmt='yuv420p', vcodec=vcodec) \
                                .overwrite_output() \
                                .global_args('-cpuflags', '0', '-loglevel', 'quiet' if not vipy.globals.isdebug() else 'debug') \
                                .run_async(pipe_stdin=True)                
                for frame in self._array:
                    process.stdin.write(frame.astype(np.uint8).tobytes())
                process.stdin.close()
                process.wait()
            
            elif (self.isdownloaded() and self._isdirty()) or isRTSPurl(self.filename()) or isRTMPurl(self.filename()):
                # Transcode the video file directly, do not load() then export
                # Requires saving to a tmpfile if the output filename is the same as the input filename
                tmpfile = '%s.tmp%s' % (filefull(outfile), fileext(outfile)) if outfile == self.filename() else outfile
                self._ffmpeg.filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                            .output(filename=tmpfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                            .overwrite_output() \
                            .global_args('-cpuflags', '0', '-loglevel', 'quiet' if not vipy.globals.isdebug() else 'debug') \
                            .run()
                if outfile == self.filename():
                    if os.path.exists(self.filename()):
                        os.remove(self.filename())
                    shutil.move(tmpfile, self.filename())
            elif self.hasfilename() and not self._isdirty():
                shutil.copyfile(self.filename(), outfile)
            elif self.hasurl() and not self.hasfilename():
                raise ValueError('Input video url "%s" not downloaded, call download() first' % self.url())
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
        return self.clone(flushforward=True, flushfilter=True, flushbackward=flush).filename(outfile).nourl()
    
    def savetmp(self):
        """Call `vipy.video.Video.saveas` using a new temporary video file, and return the video object with this new filename"""
        return self.saveas(outfile=tempMP4())
    def savetemp(self):
        """Alias for `vipy.video.Video.savetmp`"""
        return self.savetmp()

    def ffplay(self):
        """Play the video file using ffplay"""
        assert self.hasfilename() or (self.hasurl() and self.download().hasfilename())  # triggers download if needed
        cmd = 'ffplay "%s"' % self.filename()
        print('[vipy.video.play]: Executing "%s"' % cmd)
        os.system(cmd)
        return self
        
    def play(self, verbose=False, notebook=False):
        """Play the saved video filename in self.filename()

        If there is no filename, try to download it.  If the filter chain is dirty or the pixels are loaded, dump to temp video file first then play it.  This uses 'ffplay' on the PATH if available, otherwise uses a fallback player by showing a sequence of matplotlib frames.
        If the output of the ffmpeg filter chain has modified this video, then this will be saved to a temporary video file.  To play the original video (indepenedent of the filter chain of this video), use `vipy.video.Video.ffplay`.
        
        Args:
            verbose: If true, show more verbose output 
            notebook:  If true, play in a jupyter notebook

        Returns:
            The unmodified video object
        """
        

        if not self.isdownloaded() and self.hasurl():
            self.download()
        
        if notebook:
            # save to temporary video, this video is not cleaned up and may accumulate            
            try_import("IPython.display", "ipython"); import IPython.display
            if not self.hasfilename() or self.isloaded() or self._isdirty():
                v = self.saveas(tempMP4())                 
                warnings.warn('Saving video to temporary file "%s" for notebook viewer ... ' % v.filename())
                return IPython.display.Video(v.filename(), embed=True)
            return IPython.display.Video(self.filename(), embed=True)
        elif has_ffplay:
            if self.isloaded() or self._isdirty():
                f = tempMP4()
                if verbose:
                    warnings.warn('%s - Saving video to temporary file "%s" for ffplay ... ' % ('Video loaded into memory' if self.isloaded() else 'Dirty FFMPEG filter chain', f))
                v = self.saveas(f)
                cmd = 'ffplay "%s"' % v.filename()
                if verbose:
                    print('[vipy.video.play]: Executing "%s"' % cmd)
                os.system(cmd)
                if verbose:
                    print('[vipy.video.play]:  Removing temporary file "%s"' % v.filename())                    
                os.remove(v.filename())  # cleanup
            elif self.hasfilename() or (self.hasurl() and self.download().hasfilename()):  # triggers download
                self.ffplay()
            else:
                raise ValueError('Invalid video file "%s" - ffplay requires a video filename' % self.filename())
            return self

        else:
            """Fallback player.  This can visualize videos without ffplay, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates."""
            fps = self.framerate()
            assert fps > 0, "Invalid display framerate"
            with Stopwatch() as sw:            
                for (k,im) in enumerate(self.load() if self.isloaded() else self.stream()):
                    time.sleep(max(0, (1.0/self.framerate())*int(np.ceil((self.framerate()/fps))) - sw.since()))                                
                    im.show(figure=figure)
                    if vipy.globals._user_hit_escape():
                        break                    
            vipy.show.close(figure)
            return self

    def show(self):
        """Alias for play"""
        return self.play()
    
    def quicklook(self, n=9, mindim=256, startframe=0, animate=False, dt=30):
        """Generate a montage of n uniformly spaced frames.
           Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame.
        
           Input:
              -n:  Number of images in the quicklook
              -mindim:  The minimum dimension of each of the elements in the montage
              -animate:  If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames
              -dt:  The number of frames for animation
              -startframe:  The initial frame index to start the n uniformly sampled frames for the quicklook
        """
        if not self.isloaded():
            self.load()  
        if animate:
            return Video(frames=[self.quicklook(n=n, startframe=k, animate=False, dt=dt) for k in range(0, min(dt, len(self)))], framerate=self.framerate())
        framelist = [min(int(np.round(f))+startframe, len(self)-1) for f in np.linspace(0, len(self)-1, n)]
        imframes = [self.frame(k).maxmatte()  # letterbox or pillarbox
                    for (j,k) in enumerate(framelist)]
        imframes = [im.savefig(figure=1).rgb() for im in imframes]  # temp storage in memory
        return vipy.visualize.montage(imframes, imgwidth=mindim, imgheight=mindim)

    def torch(self, startframe=0, endframe=None, length=None, stride=1, take=None, boundary='repeat', order='nchw', verbose=False, withslice=False, scale=1.0, withlabel=False, nonelabel=False):
        """Convert the loaded video of shape NxHxWxC frames to an MxCxHxW torch tensor/

        Args:
            startframe: [int >= 0] The start frame of the loaded video to use for constructig the torch tensor
            endframe: [int >= 0] The end frame of the loaded video to use for constructing the torch tensor
            length: [int >= 0] The length of the torch tensor if endframe is not provided. 
            stride: [int >= 1] The temporal stride in frames.  This is the number of frames to skip.
            take: [int >= 0]  The number of uniformly spaced frames to include in the tensor.  
            boundary: ['repeat', 'cyclic'] The boundary handling for when the requested tensor slice goes beyond the end of the video
            order: ['nchw', 'nhwc', 'chwn', 'cnhw']  The axis ordering of the returned torch tensor N=number of frames (batchsize), C=channels, H=height, W=width
            verbose [bool]: Print out the slice used for contructing tensor
            withslice: [bool] Return a tuple (tensor, slice) that includes the slice used to construct the tensor.  Useful for data provenance.
            scale: [float] An optional scale factor to apply to the tensor. Useful for converting [0,255] -> [0,1]
            withlabel: [bool] Return a tuple (tensor, labels) that includes the N framewise activity labels.  
            nonelabel: [bool] returns tuple (t, None) if withlabel=False

        Returns
            Returns torch float tensor, analogous to torchvision.transforms.ToTensor()           
            Return (tensor, slice) if withslice=True (withslice takes precedence)
            Returns (tensor, labellist) if withlabel=True

        .. notes::
            - This triggers a load() of the video
            - The precedence of arguments is (startframe, endframe) or (startframe, startframe+length), then stride and take.
            - Follows numpy slicing rules.  Optionally return the slice used if withslice=True
        """
        try_import('torch'); import torch
        frames = self.load().numpy() if self.load().numpy().ndim == 4 else np.expand_dims(self.load().numpy(), 3)  # NxHxWx(C=1, C=3)
        assert boundary in ['repeat', 'strict', 'cyclic'], "Invalid boundary mode - must be in ['repeat', 'strict', 'cyclic']"

        # Slice index (i=start (zero-indexed), j=end (non-inclusive), k=step)
        (i,j,k) = (startframe, endframe, stride)
        if startframe == 'random':
            assert length is not None, "Random start frame requires fixed length"
            i = max(0, np.random.randint(len(frames)-length+1))
        if endframe is not None:
            assert length is None, "Cannot specify both endframe and length"                        
            assert endframe > startframe, "End frame must be greater than start frame"
            (j,k) = (endframe, 1)
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
        n = len(frames)  # true video length for labels
        if boundary == 'repeat' and j > len(frames):
            for d in range(j-len(frames)):
                frames = np.concatenate( (frames, np.expand_dims(frames[-1], 0) ))
        elif boundary == 'cyclic' and j > len(frames):
            for d in range(j-len(frames)):
                frames = np.concatenate( (frames, np.expand_dims(frames[j % len(frames)], 0) ))
        assert j <= len(frames), "invalid slice=%s for frame shape=%s" % (str((i,j,k)), str(frames.shape))
        if verbose:
            print('[vipy.video.torch]: slice (start,end,step)=%s for frame shape (N,C,H,W)=%s' % (str((i,j,k)), str(frames.shape)))

        # Slice and transpose to torch tensor axis ordering
        t = torch.from_numpy(frames[i:j:k] if (k!=1 or i!=0 or j!=len(frames)) else frames)  # do not copy - This shares the numpy buffer of the video, be careful!
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(-1)  # HxW -> (N=1)xHxWx(C=1)
        if order == 'nchw':
            t = t.permute(0,3,1,2)  # NxCxHxW, view
        elif order == 'nhwc':
            pass  # NxHxWxC  (native numpy order)
        elif order == 'cnhw' or order == 'cdhw':
            t = t.permute(3,0,1,2)  # CxNxHxW == CxDxHxW (for torch conv3d), view
        elif order == 'chwn':
            t = t.permute(3,1,2,0)  # CxHxWxN, view
        else:
            raise ValueError("Invalid order = must be in ['nchw', 'nhwc', 'chwn', 'cnhw']")
            
        # Scaling (optional)
        if scale is not None and self.colorspace() != 'float':
            t = (1.0/255.0)*t  # [0,255] -> [0,1]
        elif scale is not None and scale != 1.0:
            t = scale*t

        # Return tensor or (tensor, slice) or (tensor, labels)
        if withslice:
            return (t, (i,j,k))
        elif withlabel:            
            labels = [sorted(tuple(self.activitylabels( (f%n) if boundary == 'cyclic' else min(f, n-1) ))) for f in range(i,j,k)]
            return (t, labels)
        elif nonelabel:
            return (t, None)
        else:
            return t

    def clone(self, flushforward=False, flushbackward=False, flush=False, flushfilter=False, rekey=False, flushfile=False, shallow=False, sharedarray=False):
        """Create deep copy of video object, flushing the original buffer if requested and returning the cloned object.

        Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned 
        object which can be used for encoding and will be garbage collected.
        
        Args:
            flushforward: copy the object, and set the cloned object `vipy.video.Video.array` to None.  This flushes the video buffer for the clone, not the object
            flushbackward:  copy the object, and set the object array() to None.  This flushes the video buffer for the object, not the clone.
            flush:  set the object array() to None and clone the object.  This flushes the video buffer for both the clone and the object.
            flushfilter:  Set the ffmpeg filter chain to the default in the new object, useful for saving new videos
            flushfile:  Remove the filename and the URL from the video object.  Useful for creating new video objects from loaded pixels.  
            rekey: Generate new unique track ID and activity ID keys for this scene
            shallow:  shallow copy everything (copy by reference), except for ffmpeg object.  attributes dictionary is shallow copied
            sharedarray:  deep copy of everything, except for pixel buffer which is shared.  Changing the pixel buffer on self is reflected in the clone.

        Returns:
            A deepcopy of the video object such that changes to self are not reflected in the copy

        .. note:: Cloning videos is an expensive operation and can slow down real time code. Use sparingly. 

        """
        if flush or (flushforward and flushbackward):
            self._array = None  # flushes buffer on object and clone
            #self._previewhash = None
            self._shape = None
            v = copy.deepcopy(self)  # object and clone are flushed
        elif flushbackward:
            v = copy.deepcopy(self)  # propagates _array to clone
            self._array = None   # object flushed, clone not flushed
            #self._previewhash = None
            self._shape = None
        elif flushforward:
            array = self._array;
            self._array = None
            #self._previewhash = None
            self._shape = None
            v = copy.deepcopy(self)   # does not propagate _array to clone
            self._array = array    # object not flushed
            v._array = None   # clone flushed
        elif shallow:
            v = copy.copy(self)  # shallow copy
            v._ffmpeg = copy.deepcopy(self._ffmpeg)  # except for ffmpeg object
            v.attributes = {k:v for (k,v) in self.attributes.items()}  # shallow copy of keys
            v._array = np.asarray(self._array) if self._array is not None else None  # shared pixels
        elif sharedarray:
            array = self._array
            self._array = None
            v = copy.deepcopy(self)  # deep copy of everything but pixels
            v._array = np.asarray(array) if array is not None else None  # shared pixels
            self._array = array # restore
        else:
            v = copy.deepcopy(self)            

        if flushfilter:
            v._ffmpeg = ffmpeg.input(v.filename())  # no other filters
            #v._previewhash = None
            v._shape = None
            (v._startframe, v._endframe) = (None, None)
            (v._startsec, v._endsec) = (None, None)
        if rekey:
            v.rekey()
        if flushfile:
            v.nofilename().nourl()
        return v

    def flush(self):
        """Alias for clone(flush=True), returns self not clone"""
        self._array = None  # flushes buffer on object and clone
        #self._previewhash = None
        self._shape = None
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

    def gain(self, g):
        """Pixelwise multiplicative gain, such that each pixel p_{ij} = g * p_{ij}"""
        return self.normalize(0, 1, scale=g)

    def bias(self, b):
        """Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}"""
        return self.normalize(mean=0, std=1, scale=1.0, bias=b)
    
    def float(self):
        self.load()
        self._array = self._array.astype(np.float32) if self._array is not None else self._array
        return self

    def channel(self, c):
        self.load()
        assert c >= 0 and c < self.channels()
        self._array = self._array[:,:,:,c] if self._array is not None else self._array
        return self

    def normalize(self, mean, std, scale=1, bias=0):
        """Pixelwise whitening, out = ((scale*in) - mean) / std); triggers load().  All computations float32"""
        assert scale >= 0, "Invalid input"
        assert all([s > 0 for s in tolist(std)]), "Invalid input"
        self._array = vipy.math.normalize(self._array, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32), np.float32(scale))
        if bias != 0:
            self._array = self._array + np.array(bias, dtype=np.float32)
        return self.colorspace('float')

    def setattribute(self, k, v=None):
        if self.attributes is None:
            self.attributes = {}
        self.attributes[k] = v
        return self

    def hasattribute(self, k):
        return isinstance(self.attributes, dict) and k in self.attributes

    def delattribute(self, k):
        if k in self.attributes:
            self.attributes.pop(k)
        return self

    def getattribute(self, k):
        return self.attributes[k]

    
class VideoCategory(Video):
    """vipy.video.VideoCategory class

    A VideoCategory is a video with associated category, such as an activity class.  This class includes all of the constructors of vipy.video.Video 
    along with the ability to extract a clip based on frames or seconds.

    """
    def __init__(self, filename=None, url=None, framerate=30.0, attributes=None, category=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None):
        super().__init__(url=url, filename=filename, framerate=framerate, attributes=attributes, array=array, colorspace=colorspace,
                                            startframe=startframe, endframe=endframe, startsec=startsec, endsec=endsec)
        self._category = category                

    @classmethod
    def from_json(cls, s):
        d = json.loads(s) if not isinstance(s, dict) else s                        
        v = super().from_json(s)
        v._category = d['_category']
        return v
        
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))
        if self.filename() is not None:
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if not self.isloaded() and self._startframe is not None and self._endframe is None:
            strlist.append('clip=(%d,)' % (self._startframe))
        return str('<vipy.video.VideoCategory: %s>' % (', '.join(strlist)))

    def json(self, encode=True):
        d = json.loads(super().json())
        d['_category'] = self._category
        return json.dumps(d) if encode else d
    
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
        super().__init__(url=url, filename=filename, framerate=framerate, attributes=attributes, array=array, colorspace=colorspace,
                                    category=category, startframe=startframe, endframe=endframe, startsec=startsec, endsec=endsec)

        # Tracks must be defined relative to the clip specified by this constructor
        if tracks is not None:
            tracks = tracks if isinstance(tracks, list) or isinstance(tracks, tuple) else [tracks]  # canonicalize
            assert all([isinstance(t, vipy.object.Track) for t in tracks]), "Invalid track input; tracks=[vipy.object.Track(), ...]"
            self._tracks = {t.id():t for t in tracks}

        # Activities must be defined relative to the clip specified by this constructor            
        if activities is not None:
            activities = activities if isinstance(activities, list) or isinstance(activities, tuple) else [activities]  # canonicalize            
            assert all([isinstance(a, vipy.activity.Activity) for a in activities]), "Invalid activity input; activities=[vipy.activity.Activity(), ...]"
            self._activities = {a.id():a for a in activities}

        self._currentframe = None  # used during iteration only

    @classmethod
    def cast(cls, v, flush=False):
        """Cast a conformal vipy object to this class.  This is useful for downcast and upcast conversion of video objects."""
        assert isinstance(v, vipy.video.Video), "Invalid input - must be derived from vipy.video.Video"
        if v.__class__ != vipy.video.Scene:
            v.__class__ = vipy.video.Scene            
            v._tracks = {} if flush or not hasattr(v, '_tracks') else v._tracks
            v._activities = {} if flush or not hasattr(v, '_activities') else v._activities
            v._category = None if flush or not hasattr(v, '_category') else v._category
        return v

    @classmethod
    def from_json(cls, s):
        """Restore an object serialized with self.json()
        
           Usage:
           >>> vs = vipy.video.Scene.from_json(v.json())

        """

        d = json.loads(s) if not isinstance(s, dict) else s                                
        v = super().from_json(s)

        # Packed attribute storage:
        #   - When loading a large number of vipy objects, the python garbage collector slows down signficantly due to reference cycle counting
        #   - Mutable objects and custom containers are tracked by the garbage collector and the more of them that are loaded the longer GC takes
        #   - To avoid this, load attributes as tuples of packed strings.  This is an immutable type that is not refernce counted.  Check this with gc.is_tracked()
        #   - Then, unpack load the attributes on demand when accessing tracks() or activities().  Then, the nested containers are reference counted (even though they really should not since there are no cycles by construction)
        #   - This is useful when calling vipy.util.load(...) on archives that contain hundreds of thousands of objects
        #   - Do not access the private attributes self._tracks and self._attributes as they will be packed until needed
        #   - Should install ultrajson (pip install ujson) for super fast parsing
        v._tracks = tuple(d['_tracks'].values())  # efficient garbage collection: store as a packed string to avoid reference cycle tracking, unpack on demand
        v._activities = tuple(d['_activities'].values())  # efficient garbage collection: store as a packed string to avoid reference cycle tracking, unpack on demand 
        return v
        
    def pack(self):
        """Packing a scene returns the scene with the annotations JSON serialized.  
               
        - This is useful for fast garbage collection when there are many objects in memory
        - This is useful for distributed processing prior to serializing from a scheduler to a client
        - This is useful for lazy deserialization of complex attributes when loading many videos into memory
        - Unpacking is transparent to the end user and is performed on the fly when annotations are accessed.  There is no unpack() method.
        - See the notes in from_json() for why this helps with nested containers and reference cycle tracking with the python garbage collector        

        """
        d = json.loads(self.json())
        self._tracks = tuple(d['_tracks'].values())  # efficient garbage collection: store as a packed string to avoid reference cycle tracking, unpack on demand
        self._activities = tuple(d['_activities'].values())  # efficient garbage collection: store as a packed string to avoid reference cycle tracking, unpack on demand 
        return self

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d, color=%s" % (self.height(), self.width(), len(self._array), self.colorspace()))
        if self.filename() is not None:
            strlist.append('filename="%s"' % (self.filename()))
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self._framerate is not None:
            strlist.append('fps=%1.1f' % float(self._framerate))
        if not self.isloaded() and self._startframe is not None and self._endframe is not None:
            strlist.append('clip=(%d,%d)' % (self._startframe, self._endframe))
        if not self.isloaded() and self._startframe is not None and self._endframe is None:
            strlist.append('clip=(%d,)' % (self._startframe))
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.hastracks():
            strlist.append('tracks=%d' % len(self._tracks))
        if self.hasactivities():
            strlist.append('activities=%d' % len(self._activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))


    def instanceid(self, newid=None):
        """Return an annotation instance identifier for this video.  

        An instance ID is a unique identifier for a ground truth annotation within a video, either a track or an activity.  More than one instance ID may share the same video ID if they are from the same source videofile.  

        This is useful when calling `vipy.video.Scene.activityclip` or `vipy.video.Scene.activitysplit` to clip a video into segments such that each clip has a unique identifier, but all share the same underlying `vipy.video.Video.videoid`.
        This is useful when calling `vipy.video.Scene.trackclip` or `vipy.video.Scene.tracksplit` to clip a video into segments such that each clip has a unique identifier, but all share the same underlying `vipy.video.Video.videoid`.
        
        Returns:
            INSTANCEID: if 'instance_id' key is in self.attribute
            VIDEOID_INSTANCEID: if '_instance_id' key is in self.attribute, as set by activityclip() or trackclip().  This is set using INSTANCE_ID=ACTIVITYID_ACTIVITYINDEX or INSTANCEID=TRACKID_TRACKINDEX, where the index is the temporal order of the annotation in the source video prior to clip().
            VIDEOID_ACTIVITYINDEX: if 'activityindex' key is in self.attribute, as set by activityclip().  (fallback for legacy datasets).
            VIDEOID: otherwise 
        """
        if newid is not None:
            self.setattribute('instance_id', newid)
            return self
        else:
            if 'instance_id' in self.attributes:
                return self.attributes['instance_id']
            elif '_instance_id' in self.attributes:
                return '%s_%s' % (self.videoid(), str(self.attributes['_instance_id']))
            elif 'activityindex' in self.attributes:
                return '%s_%s' % (self.videoid(), str(self.attributes['activityindex']))
            else:
                return self.videoid()

    def frame(self, k, img=None):
        """Return `vipy.image.Scene` object at frame k

        -The attributes of each of the `vipy.image.Scene.objects` in the scene contains helpful metadata for the provenance of the detection, including:  
            - 'trackid' of the track this detection
            - 'activityid' associated with this detection 
            - 'jointlabel' of this detection, used for visualization
            - 'noun verb' of this detection, used for visualization
        
        Args:
            k: [int >=- 0] The frame index requested.  This is relative to the current frame rate of the video.
            img: [numpy]  An optional image to be used for this frame.  This is useful to construct frames efficiently for videos if the pixel buffer is available from a stream rather than a preview.
        
        Return:
            A `vipy.image.Scene` object for frame k containing all objects in this frame
        
        .. notes::
            -Modifying this frame will not affect the source video

        """
        assert isinstance(k, int) and k>=0, "Frame index must be non-negative integer"
        assert img is not None or (self.isloaded() and k<len(self)) or not self.isloaded(), "Invalid frame index %d - Indexing video by frame must be integer within (0, %d)" % (k, len(self)-1)

        img = img if img is not None else (self._array[k] if self.isloaded() else self.preview(k).array())
        dets = [t[k].clone(deep=True).setattribute('trackindex', j) for (j, t) in enumerate(self.tracks().values()) if len(t)>0 and (t.during(k) or t.boundary()=='extend')]  # track interpolation (cloned) with boundary handling
        for d in dets:
            d.attributes['activityid'] = []  # reset
            jointlabel = [(d.shortlabel(),'')]  # [(Noun, Verbing1), (Noun, Verbing2), ...]
            activityconf = [None]   # for display 

            for (aid, a) in self.activities().items():  # insertion order:  First activity is primary, next is secondary (not in confidence order) 
                if a.hastrack(d.attributes['trackid']) and a.during(k):
                    # Jointlabel is always displayed as "Noun Verbing" during activity (e.g. Person Carrying, Vehicle Turning) using noun=track shortlabel, verb=activity shortlabel
                    # If noun is associated with more than one activity, then this is shown as "Noun Verbing1\nNoun Verbing2", with a newline separator 
                    if not any([a.shortlabel() == v for (n,v) in jointlabel]):
                        jointlabel.append( (d.shortlabel(), a.shortlabel()) )  # only show each activity once (even if repeated)
                        activityconf.append(a.confidence())
                    d.attributes['activityid'].append(a.id())  # for activity correspondence (if desired)

            # For display purposes
            d.attributes['jointlabel'] = '\n'.join([('%s %s' % (n,v)).strip() for (n,v) in jointlabel[0 if len(jointlabel)==1 else 1:]])
            d.attributes['noun verb'] = jointlabel[0 if len(jointlabel)==1 else 1:]
            d.attributes['activityconf'] = activityconf[0 if len(jointlabel)==1 else 1:]
        dets.sort(key=lambda d: (d.confidence() if d.confidence() is not None else 0, d.shortlabel()))   # layering in video is ordered by decreasing track confidence and alphabetical shortlabel
        return vipy.image.Scene(array=img, colorspace=self.colorspace(), objects=dets, category=self.category())  
                
        
    def during(self, frameindex):
        try:
            self.__getitem__(frameindex)  # triggers load
            return True
        except:
            return False
            
    def labeled_frames(self):
        """Iterate over frames, yielding tuples (activity+object labelset in scene, vipy.image.Scene())"""
        self.load()
        for k in range(0, len(self)):
            self._currentframe = k    # used only for incremental add()
            yield (self.labels(k), self.__getitem__(k))
        self._currentframe = None
        

    def framecomposite(self, n=2, dt=10, mindim=256):
        """Generate a single composite image with minimum dimension mindim as the uniformly blended composite of n frames each separated by dt frames"""
        if not self.isloaded():
            self.mindim(mindim).load()
        imframes = [self.frame(k).maxmatte() for k in range(0, dt*n, dt)]
        img = np.uint8(np.sum([1/float(n)*im.array() for im in imframes], axis=0))
        return imframes[0].clone().array(img)

    def isdegenerate(self):
        """Degenerate scene has empty or malformed tracks"""
        return len(self.tracklist()) == 0 or any([t.isempty() or t.isdegenerate() for t in self.tracklist()])
    
    def quicklook(self, n=9, dilate=1.5, mindim=256, fontsize=10, context=False, startframe=0, animate=False, dt=30):
        """Generate a montage of n uniformly spaced annotated frames centered on the union of the labeled boxes in the current frame to show the activity ocurring in this scene at a glance
           Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame.  This quicklook is most useful when len(self.activities()==1)
           for generating a quicklook from an activityclip().
        
           Input:
              -n:  Number of images in the quicklook
              -dilate:  The dilation factor for the bounding box prior to crop for display
              -mindim:  The minimum dimension of each of the elemnets in the montage
              -fontsize:  The size of the font for the bounding box label
              -context:  If true, replace the first and last frame in the montage with the full frame annotation, to help show the scale of the scene
              -animate:  If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames
              -dt:  The number of frames for animation
              -startframe:  The initial frame index to start the n uniformly sampled frames for the quicklook
        """
        if not self.isloaded():
            self.load()  
        if animate:
            return Video(frames=[self.quicklook(n=n, dilate=dilate, mindim=mindim, fontsize=fontsize, context=context, startframe=k, animate=False, dt=dt) for k in range(0, min(dt, len(self)))], framerate=self.framerate())
        f_mutator = vipy.image.mutator_show_jointlabel()
        framelist = [min(int(np.round(f))+startframe, len(self)-1) for f in np.linspace(0, len(self)-1, n)]
        isdegenerate = [self.frame(k).boundingbox() is None or self.frame(k).boundingbox().dilate(dilate).intersection(self.framebox(), strict=False).isdegenerate() for (j,k) in enumerate(framelist)]
        imframes = [self.frame(k).maxmatte()  # letterbox or pillarbox
                    if (isdegenerate[j] or (context is True and (j == 0 or j == (n-1)))) else
                    self.frame(k).padcrop(self.frame(k).boundingbox().dilate(dilate).imclipshape(self.width(), self.height()).maxsquare().int()).mindim(mindim, interp='nearest')
                    for (j,k) in enumerate(framelist)]
        imframes = [f_mutator(im) for im in imframes]  # show jointlabel from frame interpolation
        imframes = [im.savefig(fontsize=fontsize, figure=1).rgb() for im in imframes]  # temp storage in memory
        return vipy.visualize.montage(imframes, imgwidth=mindim, imgheight=mindim)
    
    def tracks(self, tracks=None, id=None):
        """Return mutable dictionary of tracks"""        
        if isinstance(self._tracks, tuple):
            self._tracks = {t.id():t for t in [vipy.object.Track.from_json(json.loads(s)) for s in self._tracks]}  # on-demand unpack (efficient garbage collection for large list of objects)
        if tracks is None and id is None:
            return self._tracks  # mutable dict
        elif id is not None:
            return self._tracks[id]
        elif isinstance(tracks, dict):
            assert all([isinstance(t, vipy.object.Track) and k == t.id() for (k,t) in tracks.items()]), "Invalid input - Must be dictionary of vipy.object.Track"
            self._tracks = tracks.copy()  # shallow copy
            return self
        else:
            assert all([isinstance(t, vipy.object.Track) for t in tolist(tracks)]), "Invalid input - Must be vipy.object.Track or list of vipy.object.Track"
            self._tracks = {t.id():t for t in tolist(tracks)}  # insertion order preserved (python >=3.6)
            return self

    def track(self, id):
        return self.tracks(id=id)

    def trackindex(self, id):
        assert id in self.tracks()
        return [t.id() for t in self.tracklist()].index(id)

    def trackidx(self, idx):
        return self.tracklist()[idx]

    def activity(self, id):
        return self.activities(id=id)
        
    def next_activity(self, id):
        """Return the next activity just after the given activityid"""
        assert id in self.activities()
        A = self.activitylist()
        k = [k for (k,a) in enumerate(A) if a.id() == id][0]
        return A[k+1] if k<len(A)-1 else None

    def prev_activity(self, id):
        """Return the previous activity just before the given activityid"""
        assert id in self.activities()
        A = self.activitylist()
        k = [k for (k,a) in enumerate(A) if a.id() == id][0]
        return A[k-1] if k>=1 else None

    def tracklist(self):
        return list(self.tracks().values())  # triggers copy

    def actorid(self, id=None, fluent=False):
        """Return or set the actor ID for the video.

        - The actor ID is the track ID of the primary actor in the scene.  This is useful for assigning a role for activities that are performed by the actor.
        - The actor ID is the first track is in the tracklist
        
        Args:
            id: [str] if not None, then use this track ID as the actor
            fluent: [bool] If true, always return self. This is useful for those cases where the actorid being set is None.
        
        Returns:
            [id=None, fluent=False] the actor ID
            [id is not None] The video with the actor ID set.        
        """
        if id is None:
            return next(iter(self.tracks().keys())) if not fluent else self  # Python >=3.6
        else:
            # Reorder tracks so that id is first
            assert id in self.tracks()
            idlist = [id] + [ti for ti in self.tracks().keys() if ti != id]
            self._tracks = {k:self.track(k) for k in idlist}
            return self

    def setactorid(self, id):
        """Alias for `vipy.video.Scene.actorid`"""
        return self.actorid(id, fluent=True)

    def actor(self):
        #assert len(self.tracks()) == 1, "Actor only valid for scenes with a single track"
        return next(iter(self.tracks().values()))   # Python >=3.6
        
    def primary_activity(self):
        """Return the primary activity of the video.

        - The primary activity is the first activity in the activitylist.  
        - This is useful for activityclip() videos that are centered on a single activity
        
        Returns:
            `vipy.activity.Activity` that is first in the `vipy.video.Scene.activitylist`
        """
        return next(iter(self.activities().values())) if len(self._activities)>0 else None  # Python >=3.6        

    def activities(self, activities=None, id=None):
        """Return mutable dictionary of activities.  All temporal alignment is relative to the current clip()."""
        if isinstance(self._activities, tuple):
            self._activities = {a.id():a for a in [vipy.activity.Activity.from_json(json.loads(s)) for s in self._activities]}  # on-demand
        if activities is None and id is None:
            return self._activities  # mutable dict
        elif id is not None:
            return self._activities[id]
        elif isinstance(activities, dict):
            assert all([isinstance(a, vipy.activity.Activity) and k == a.id() for (k,a) in activities.items()]), "Invalid input - Must be dictionary of vipy.activity.Activity"
            self._activities = activities.copy()  # shallow copy
            return self
        else:
            assert all([isinstance(a, vipy.activity.Activity) for a in tolist(activities)]), "Invalid input - Must be vipy.activity.Activity or list of vipy.activity.Activity"
            self._activities = {a.id():a for a in tolist(activities)}   # insertion order preserved (python >=3.6)
            return self

    def activityindex(self, k):
        alist = self.activitylist()
        assert k >= 0 and k < len(alist), "Invalid index"        
        return alist[k]

    def activitylist(self):
        return list(self.activities().values())  # insertion ordered (python >=3.6), triggers copy
        
    def activityfilter(self, f):
        """Apply boolean lambda function f to each activity and keep activity if function is true, remove activity if function is false
        
        Filter out all activities longer than 128 frames 
        >>> vid = vid.activityfilter(lambda a: len(a)<128)

        Filter out activities with category in set
        >>> vid = vid.activityfilter(lambda a: a.category() in set(['category1', 'category2']))
       
        Args:
            f: [lambda] a lambda function that takes an activity and returns a boolean

        Returns:
            This video with the activities f(a)==False removed.

        """
        assert callable(f)
        self._activities = {k:a for (k,a) in self.activities().items() if f(a) == True}
        return self
        
    def trackfilter(self, f, activitytrack=True):
        """Apply lambda function f to each object and keep if filter is True.  
        
        Args:
            activitytrack: [bool] If true, remove track assignment from activities also, may result in activities with no tracks
            f: [lambda]  The lambda function to apply to each track t, and if f(t) returns True, then keep the track
        
        Returns:
            self, with tracks removed in-place

        .. note:: Applying track filter with activitytrack=True may result in activities with no associated tracks.  You should follow up with self.activityfilter(lambda a: len(a.trackids()) > 0).
        """
        assert callable(f)
        self._tracks = {k:t for (k,t) in self.tracks().items() if f(t) == True}
        if activitytrack:
            self.activitymap(lambda a: a.trackfilter(lambda ti: ti in self._tracks))  # remove track association in activities
            #if any([len(a.tracks()) == 0 for a in self.activitylist()]):
            #    warnings.warn('trackfilter(..., activitytrack=True) removed tracks which returned at least one degenerate activity with no tracks')
        return self

    def trackmap(self, f, strict=True):
        """Apply lambda function f to each activity

           -strict=True: enforce that lambda function must return non-degenerate Track() objects        
        """
        assert callable(f)
        self._tracks = {k:f(t) for (k,t) in self.tracks().items()}
        assert all([isinstance(t, vipy.object.Track) and (strict is False or not t.isdegenerate()) for (tk,t) in self.tracks().items()]), "Lambda function must return non-degenerate vipy.object.Track()"
        return self
        
    def activitymap(self, f):
        """Apply lambda function f to each activity"""
        assert callable(f)
        self._activities = {k:f(a) for (k,a) in self.activities().items()}
        assert all([isinstance(a, vipy.activity.Activity) for a in self.activitylist()]), "Lambda function must return vipy.activity.Activity()"
        return self

    def rekey(self):
        """Change the track and activity IDs to randomly assigned UUIDs.  Useful for cloning unique scenes"""
        d_old_to_new = {k:hex(int(uuid.uuid4().hex[0:8], 16))[2:] for (k,a) in self.activities().items()}
        self._activities = {d_old_to_new[k]:a.id(d_old_to_new[k]) for (k,a) in self.activities().items()}
        d_old_to_new = {k:hex(int(uuid.uuid4().hex[0:8], 16))[2:] for (k,t) in self.tracks().items()}
        self._tracks = {d_old_to_new[k]:t.id(d_old_to_new[k]) for (k,t) in self.tracks().items()}
        for (k,v) in d_old_to_new.items():
            self.activitymap(lambda a: a.replaceid(k,v) )
        return self

    def label(self):
        """Return an iterator over labels in each frame"""
        endframe = max([a.endframe() for a in self.activitylist()]+[t.endframe() for (tk,t) in self.tracks().items()])
        for k in range(0,endframe):
            yield self.labels(k)
    
    def labels(self, k=None):
        """Return a set of all object and activity labels in this scene, or at frame int(k)"""
        return self.activitylabels(k).union(self.objectlabels(k))

    def activitylabel(self, startframe=None, endframe=None):
        """Return an iterator over activity labels in each frame, starting from startframe and ending when there are no more activities"""        
        endframe = endframe if endframe is not None else (max([a.endframe() for a in self.activitylist()]) if len(self.activities())>0 else 0)
        startframe = startframe if startframe is not None else (min([a.startframe() for a in self.activitylist()]) if len(self.activities())>0 else 0)
        assert startframe <= endframe
        for k in range(startframe, endframe):
            yield self.activitylabels(k)
        
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
    
    def objectlabels(self, k=None, lower=False):
        """Return a python set of all activity categories in this scene, or at frame k.
        
        Args:
            k: [int] The object labels present at frame k.  If k=None, then all object labels in the video
            lower: [bool] If true, return the object labels in alll lower case for case invariant string comparisonsn            
        """
        return set([t.category() if not lower else t.category().lower() for t in self.tracks().values() if k is None or t.during(k)])        

    def categories(self):
        """Alias for labels()"""
        return self.labels()
    
    def activity_categories(self):
        """Alias for activitylabels()"""
        return self.activitylabels()        
        
    def hasactivities(self):
        """Does this video have any activities?"""
        return len(self._activities) > 0

    def hastracks(self):
        """Does this video have any tracks?"""
        return len(self._tracks) > 0

    def hastrack(self, trackid):
        """Does the video have this trackid?  
        
        .. note:: Track IDs are available as vipy.object.Track().id()
        """
        return trackid in self.tracks()

    def add(self, obj, category=None, attributes=None, rangecheck=True, frame=None, fluent=False):
        """Add the object obj to the scene, and return an index to this object for future updates
        
        This function is used to incrementally build up a scene frame by frame.  Obj can be one of the following types:

        - obj = vipy.object.Detection(), this must be called from within a frame iterator (e.g. for im in video) to get the current frame index
        - obj = vipy.object.Track()  
        - obj = vipy.activity.Activity()
        - obj = [xmin, ymin, width, height], with associated category kwarg, this must be called from within a frame iterator to get the current frame index
        
        It is recomended that the objects are added as follows.  For a v=vipy.video.Scene():
            
        >>>    for im in v:
        >>>        # Do some processing on frame im to detect objects
        >>>        (object_labels, xywh) = object_detection(im)
        >>>
        >>>        # Add them to the scene, note that each object instance is independent in each frame, use tracks for object correspondence
        >>>        for (lbl,bb) in zip(object_labels, xywh):
        >>>            v.add(bb, lbl)
        >>>
        >>>        # Do some correspondences to track objects
        >>>        t2 = v.add( vipy.object.Track(...) )
        >>>
        >>>        # Update a previous track to add a keyframe
        >>>        v.track(t2).add( ... )
        
        The frame iterator will keep track of the current frame in the video and add the objects in the appropriate place.  Alternatively,

        >>>    v.add(vipy.object.Track(..), frame=k)

        Args:
            obj: A conformal python object to add to the scene (`vipy.object.Detection`, `vipy.object.Track`, `vipy.activity.Activity`, [xmin, ymin, width, height]
            category:  Used if obj is an xywh tuple
            attributes:  Used only if obj is an xywh tuple
            frame:  [int] The frame to add the object
            rangecheck: [bool] If true, check if the object is within the image rectangle and throw an exception if not.  This requires introspecting the video shape using `vipy.video.Video.shape`.
            fluent: [bool] If true, return self instead of the object index 

        """        
        if isinstance(obj, vipy.object.Detection):
            assert frame is not None or self._currentframe is not None, "add() for vipy.object.Detection() must be added during frame iteration (e.g. for im in video: )"
            k = frame if frame is not None else self._currentframe
            if obj.hasattribute('trackid') and obj.attributes['trackid'] in self.tracks():
                # The attribute "trackid" is set for a detection when interpolating a track at a frame.  This is useful for reconstructing a track from previously enumerated detections
                trackid = obj.attributes['trackid']
                self.trackmap(lambda t: t.update(k, obj) if obj.attributes['trackid'] == t.id() else t) 
                return None if not fluent else self
            else:
                t = vipy.object.Track(category=obj.category(), keyframes=[k], boxes=[obj], boundary='strict', attributes=obj.attributes, trackid=obj.attributes['trackid'] if obj.hasattribute('trackid') else None)
                if rangecheck and not obj.hasoverlap(width=self.width(), height=self.height()):
                    raise ValueError("Track '%s' does not intersect with frame shape (%d, %d)" % (str(t), self.height(), self.width()))
                self.tracks()[t.id()] = t  # by-reference
                return t.id() if not fluent else self
        elif isinstance(obj, vipy.object.Track):
            if rangecheck and not obj.boundingbox().isinside(vipy.geometry.imagebox(self.shape())):
                obj = obj.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('[vipy.video.add]: Clipping trackid=%s track="%s" to image rectangle' % (str(obj.id()), str(obj)))
            if obj.framerate() != self.framerate():
                obj.framerate(self.framerate())  # convert framerate of track to framerate of video
            self.tracks()[obj.id()] = obj  # by-reference
            return obj.id() if not fluent else self
        elif isinstance(obj, vipy.activity.Activity):
            if rangecheck and obj.startframe() >= obj.endframe():
                raise ValueError("Activity '%s' has invalid (startframe, endframe)=(%d, %d)" % (str(obj), obj.startframe(), obj.endframe()))
            self.activities()[obj.id()] = obj  # by-reference, activity may have no tracks
            return obj.id() if not fluent else self
        elif (istuple(obj) or islist(obj)) and len(obj) == 4 and isnumber(obj[0]):
            assert self._currentframe is not None, "add() for obj=xywh must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=category, keyframes=[self._currentframe], boxes=[vipy.geometry.BoundingBox(xywh=obj)], boundary='strict', attributes=attributes)
            if rangecheck and not t.boundingbox().isinside(vipy.geometry.imagebox(self.shape())):
                t = t.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('Clipping track "%s" to image rectangle' % (str(t)))
            self.tracks()[t.id()] = t  # by-reference
            return t.id() if not fluent else self
        else:
            raise ValueError('Undefined object type "%s" to be added to scene - Supported types are obj in ["vipy.object.Detection", "vipy.object.Track", "vipy.activity.Activity", "[xmin, ymin, width, height]"]' % str(type(obj)))

    def delete(self, id):
        """Delete a given track or activity by id, if present"""
        return self.trackfilter(lambda t: t.id() != id).activityfilter(lambda a: a.id() != id)
            
    def addframe(self, im, frame=None):
        """Add im=vipy.image.Scene() into vipy.video.Scene() at given frame. The input image must have been generated using im=self[k] for this to be meaningful, so that trackid can be associated"""
        assert isinstance(im, vipy.image.Scene), "Invalid input - Must be vipy.image.Scene()"
        assert frame is not None or self._currentframe is not None, "Must provide a frame number"
        assert im.shape() == self.shape(), "Frame input (shape=%s) must be same shape as video (shape=%s)" % (str(im.shape()), str(self.shape()))
        
        # Copy framewise vipy.image.Scene() into vipy.video.Scene(). 
        frame = frame if frame is not None else self._currentframe  # if iterator        
        self.numpy()[frame] = im.array()  # will trigger copy        
        for bb in im.objects():
            self.trackmap(lambda t: t.update(frame, bb) if bb.attributes['trackid'] == t.id() else t) 
        return self
    
    def clear(self):
        """Remove all activities and tracks from this object"""
        self._activities = {}
        self._tracks = {}
        return self

    def cleartracks(self):
        self._tracks = {}
        return self

    def clearactivities(self):
        self._activities = {}
        return self
    
    def replace(self, other, frame=None):
        """Replace tracks and activities with other if activity/track is during frame"""
        assert isinstance(other, vipy.video.Scene)
        self.activities([a for a in other.activitylist() if frame is None or a.during(frame)])
        self.tracks([t for t in other.tracklist() if frame is None or t.during(frame)])
        return self

    def json(self, encode=True):
        """Return JSON encoded string of this object.  This may fail if attributes contain non-json encodeable object"""
        try:
            json.loads(json.dumps(self.attributes))  # rount trip for the attributes ductionary - this can be any arbitrary object and contents may not be json encodable
        except:
            raise ValueError('Video contains non-JSON encodable object in self.attributes dictionary - Try to clear with self.attributes = {} first')

        d = json.loads(super().json())
        d['_tracks'] = {k:t.json(encode=True) for (k,t) in self.tracks().items()}
        d['_activities'] = {k:a.json(encode=True) for (k,a) in self.activities().items()}
        try:
            return json.dumps(d) if encode else d
        except:
            # Legacy support for non JSON serializable objects (<= vipy.1.9.2)
            v = self.clone()
            for (ti, t) in v.tracks().items():
                for o in t._keyboxes:
                    vipy.geometry.BoundingBox.cast(o, flush=True)
                    o.float().significant_digits(2)

            for (ai, a) in v.activities().items():
                a._startframe = int(a._startframe)
                a._endframe = int(a._endframe)
            return v.json(encode=encode)
        
    def csv(self, outfile=None):
        """Export scene to CSV file format with header.  If there are no tracks, this will be empty. """
        assert self.load().isloaded()
        csv = [(self.filename(), # video filename
                k,  # frame number (zero indexed)
                d.category(), d.shortlabel(), # track category and shortlabel (displayed in caption)
                ';'.join([self.activities(id=aid).category() for aid in tolist(d.attributes['activityid'])] if 'activityid' in d.attributes else ''), # semicolon separated activity category associated with track
                d.xmin(), d.ymin(), d.width(), d.height(),   # bounding box
                d.attributes['trackid'],  # globally unique track ID
                ';'.join([aid for aid in tolist(d.attributes['activityid'])] if 'activityid' in d.attributes else '')) # semicolon separated activity ID associated with track
               for (k,im) in enumerate(self) for d in im.objects()]
        csv = [('# video_filename', 'frame_number', 'object_category', 'object_shortlabel', 'activity categories(;)', 'xmin', 'ymin', 'width', 'height', 'track_id', 'activity_ids(;)')] + csv
        return writecsv(csv, outfile) if outfile is not None else csv


    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations.

        >>> fps = self.framerate()
        >>> self.framerate(fps=15.0)

        """
        
        if fps is None:
            return self._framerate
        elif fps == self._framerate:
            return self
        else:
            assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
            self._startframe = int(round(self._startframe * (fps/self._framerate))) if self._startframe is not None else self._startframe  # __repr__ only
            self._endframe = int(round(self._endframe * (fps/self._framerate))) if self._endframe is not None else self._endframe  # __repr__only
            self._tracks = {k:t.framerate(fps) for (k,t) in self.tracks().items()}
            self._activities = {k:a.framerate(fps) for (k,a) in self.activities().items()}        
            if 'fps=' in self._ffmpeg_commandline():
                self._update_ffmpeg('fps', fps)  # replace fps filter, do not add to it
            else:
                self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')  # create fps filter first time
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
        return [vid.clone().setattribute('activityindex', k).activities(pa).tracks(t).setactorid(pa.actorid()) for (k,(pa,t)) in enumerate(zip(activities, tracks))]

    def tracksplit(self):
        """Split the scene into k separate scenes, one for each track.  Each scene starts at frame 0 and is a shallow copy of self containing exactly one track.  

        - This is useful for visualization by breaking a scene into a list of scenes that contain only one track.
        - The attribute '_trackindex' is set in the attributes dictionary to provide provenance for the track relative to the source video

        .. notes:: Use clone() to create a deep copy if needed.
        """
        return [self.clone(shallow=True).setattribute('_trackindex', k).tracks(t).activityfilter(lambda a: a.hastrack(tk)) for (k,(tk,t)) in enumerate(self.tracks().items())]

    def trackclip(self):
        """Split the scene into k separate scenes, one for each track.  Each scene starts and ends when the track starts and ends"""
        return [t.setattribute('_instance_id', '%s_%d' % (t.actorid(), k)).clip(t.track(t.actorid()).startframe(), t.track(t.actorid()).endframe()) for (k,t) in enumerate(self.tracksplit())]
    
    def activityclip(self, padframes=0, multilabel=True):
        """Return a list of `vipy.video.Scene` objects each clipped to be temporally centered on a single activity, with an optional padframes before and after.  

        Args:
            padframes: [int] for symmetric padding same before and after
            padframes: [tuple] (int, int) for asymmetric padding before and after
            padframes: [list[tuples]] [(int, int), ...] for activity specific asymmetric padding
            multilabel: [bool] include overlapping multilabel secondary activities in each activityclip

        Returns:
            A list of `vipy.video.Scene` each cloned from the source video and clipped on one activity in the scene

        .. notes::
           - The Scene() category is updated to be the activity category of the clip, and only the objects participating in the activity are included.
           - Clips are returned ordered in the temporal order they appear in the video.
           - The returned vipy.video.Scene() objects for each activityclip are clones of the video, with the video buffer flushed.
           - Each activityclip() is associated with each activity in the scene, and includes all other secondary activities that the objects in the primary activity also perform (if multilabel=True).  See activityclip().labels(). 
           - Calling activityclip() on activityclip(multilabel=True) will duplicate activities, due to the overlapping secondary activities being included in each clip with an overlap.  Be careful!
        """
        assert isinstance(padframes, int) or istuple(padframes) or islist(padframes)

        vid = self.clone(flushforward=True)
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity clips with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))
        primary_activities = sorted([a.clone() for a in vid.activities().values() if (a.endframe()-a.startframe()) > 0], key=lambda a: a.startframe())   # only activities with at least one frame, sorted in temporal order
        padframelist = [padframes if istuple(padframes) else (padframes, padframes) for k in range(len(primary_activities))] if not islist(padframes) else padframes                    
        tracks = [ [t.clone() for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in primary_activities]  # tracks associated with each primary activity (may be empty)
        secondary_activities = [[sa.clone() for sa in primary_activities if (sa.id() != pa.id() and pa.clone().temporalpad((prepad, postpad)).hasoverlap(sa) and (len(T)==0 or any([sa.hastrack(t) for t in T])))] for (pa, T, (prepad,postpad)) in zip(primary_activities, tracks, padframelist)]  # overlapping secondary activities that includes any track in the primary activity
        secondary_activities = [sa if multilabel else [] for sa in secondary_activities]  
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        return [vid.clone()
                .activities([pa]+sa)  # primary activity first
                .tracks(t)
                .clip(startframe=max(pa.startframe()-prepad, 0), endframe=(pa.endframe()+postpad))
                .category(pa.category())
                .setactorid(pa.actorid())  # actor is actor of primary activity
                .setattribute('activityindex',k).setattribute('_instance_id', '%s_%d' % (pa.id(), k))
                for (k,(pa,sa,t,(prepad,postpad))) in enumerate(zip(primary_activities, secondary_activities, tracks, padframelist))]

    def noactivityclip(self, label=None, strict=True, padframes=0):
        """Return a list of vipy.video.Scene() each clipped on a track segment that has no associated activities.  

        Args:
            strict: [bool] True means that background can only occur in frames where no tracks are performing any activities.  This is useful so that background is not constructed from secondary objects. False means that background can only occur in frames where a given track is not performing any activities. 
            label: [str] The activity label to give the background activities.  Defaults to the track category (lowercase)
            padframes: [int]  The amount of temporal padding to apply to the clips before and after in frames.  See `vipy.video.Scene.activityclip` for options.
        
        Returns:
            A list of `vipy.video.Scene` each cloned from the source video and clipped in the temporal region between activities.  The union of activityclip() and noactivityclip() should equal the entire video.

        .. notes::
            - Each clip will contain exactly one activity "Background" which is the interval for this track where no activities are occurring
            - Each clip will be at least one frame long
        """
        v = self.clone()
        for t in v.tracklist():
            bgframe = [k for k in range(t.startframe(), t.endframe()) if not any([a.hastrack(t) and a.during(k) for a in self.activitylist()])]                
            while len(bgframe) > 0:
                (i,j) = (0, np.argwhere(np.diff(bgframe) > 1).flatten()[0] + 1 if len(np.argwhere(np.diff(bgframe) > 1))>0 else len(bgframe)-1)
                if i < j:
                    v.add(vipy.activity.Activity(label=t.category() if label is None else label, 
                                                 shortlabel='' if label is None else label,
                                                 startframe=bgframe[i], endframe=bgframe[j],
                                                 actorid=t.id(), framerate=v.framerate(), attributes={'noactivityclip':True}))
                bgframe = bgframe[j+1:]
        return v.activityfilter(lambda a: 'noactivityclip' in a.attributes).activityclip(padframes=padframes, multilabel=False)

    def trackbox(self, dilate=1.0):
        """The trackbox is the union of all track bounding boxes in the video, or None if there are no tracks
        
        Args:
            dilate: [float] A dilation factor to apply to the trackbox before returning.  See `vipy.geometry.BoundingBox.dilate`

        Returns:
            A `vipy.geometry.BoundingBox` which is the union of all boxes in the track (or None if no boxes exist)
        """
        boxes = [t.clone().boundingbox() for t in self.tracklist()]
        boxes = [bb.dilate(dilate) for bb in boxes if bb is not None]
        return boxes[0].union(boxes[1:]) if len(boxes) > 0 else None

    def framebox(self):
        """Return the bounding box for the image rectangle.

        Returns:
            A `vipy.geometry.BoundingBox` which defines the image rectangle

        .. notes: This requires calling `vipy.video.Video.preview` to get the frame shape from the current filter chain, which touches the video file"""
        return vipy.geometry.BoundingBox(xmin=0, ymin=0, width=self.width(), height=self.height())

    def trackcrop(self, dilate=1.0, maxsquare=False, zeropad=True):
        """Return the trackcrop() of the scene which is the crop of the video using the `vipy.video.Scene.trackbox`.
         
        Args:
            zeropad: [bool] If True, the zero pad the crop if it is outside the image rectangle, otherwise return only valid pixels inside the image rectangle
            maxsquare: [bool] If True, make the bounding box the maximum square before cropping
            dilate: [float] The dilation factor to apply to the trackbox prior to cropping
        
        Returns:
           A `vipy.video.Scene` object from cropping the video using the trackbox.  If there are no tracks, return None.  

        """
        bb = self.trackbox(dilate)  # may be None if trackbox is degenerate
        return self.crop(bb.maxsquareif(maxsquare), zeropad=zeropad) if bb is not None else None  

    def activitybox(self, activityid=None, dilate=1.0):
        """The activitybox is the union of all activity bounding boxes in the video, which is the union of all tracks contributing to all activities.  This is most useful after activityclip().
           The activitybox is the smallest bounding box that contains all of the boxes from all of the tracks in all activities in this video.
        """
        activities = [a for (k,a) in self.activities().items() if (activityid is None or k in set(activityid))]
        boxes = [t.clone().boundingbox().dilate(dilate) for t in self.tracklist() if any([a.hastrack(t) for a in activities])]
        return boxes[0].union(boxes[1:]) if len(boxes) > 0 else vipy.geometry.BoundingBox(xmin=0, ymin=0, width=int(self.width()), height=int(self.height()))

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
                       for (k,(ti,t)) in enumerate(self.tracks().items())}  # replace tracks with boxes relative to tube
        return vid.array(np.stack([im.numpy() for im in frames]))

    def actortube(self, trackid=None, dilate=1.0, maxdim=256, strict=True):
        """The actortube() is a sequence of crops where the spatial box changes on every frame to track the primary actor performing an activity.  
           The box in each frame is the square box centered on the primary actor performing the activity, dilated by a given factor (the original box around the actor is unchanged, this just increases the context, with zero padding)
           This function does not perform any temporal clipping.  Use activityclip() first to split into individual activities.  
           All crops will be resized so that the maximum dimension is maxdim (and square by default)
        """
        assert trackid is not None or len(self.tracks()) == 1, "Track ID must be provided if there exists more than one track in the scene"
        trackid = trackid if trackid is not None else list(self.tracks().keys())[0]
        assert self.hastrack(trackid), "Track ID %s not found - Actortube requires a track ID in the scene (tracks=%s)" % (str(trackid), str(self.tracks()))
        vid = self.clone().load()  # triggers load        
        t = vid.tracks(id=trackid)  # actor track
        frames = [im.padcrop(t[k].maxsquare().dilate(dilate).int()).resize(maxdim, maxdim) for (k,im) in enumerate(vid) if t.during(k)] if len(t)>0 else []  # actor interpolation, padding may introduce frames with no tracks
        if len(frames) == 0:
            if not strict:
                warnings.warn('[vipy.video.actortube]: Empty track for trackid="%s" - Setting actortube to zero' % trackid)
                frames = [ vid[0].resize(maxdim, maxdim).zeros() ]  # empty frame
                vid.attributes['actortube'] = {'empty':True}   # provenance to reject             
            else:
                raise ValueError('[vipy.video.actortube]: Empty track for track=%s, trackid=%s' % (str(t), trackid))
        vid._tracks = {ti:vipy.object.Track(keyframes=[f for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],  # keyframes zero indexed, relative to [frames]
                                            boxes=[d for (f,im) in enumerate(frames) for d in im.objects() if d.attributes['trackid'] == ti],  # one box per frame
                                            category=t.category(), trackid=ti)  # preserve trackid
                       for (k,(ti,t)) in enumerate(self.tracks().items())}  # replace tracks with interpolated boxes relative to tube defined by actor
        return vid.array(np.stack([im.numpy() for im in frames]))

    def speed(self, s):
        """Change the speed by a multiplier s.  If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)"""        
        super().speed(s)
        return self.trackmap(lambda t: t.framerate(speed=s)).activitymap(lambda a: a.framerate(speed=s))
        

    def clip(self, startframe, endframe=None):
        """Clip the video to between (startframe, endframe).  This clip is relative to clip() shown by __repr__(). 

        Args:
            startframe: [int] the start frame relative to the video framerate() for the clip
            endframe: [int] the end frame relative to the video framerate for the clip, may be none
        
        Returns:
            This video object, clipped so that a load() will result in frame=0 equivalent to startframe.  All tracks and activities updated relative to the new startframe.

        .. notes:  
            - This return a clone of the video for idempotence
            - This does not load the video.  This updates the ffmpeg filter chain to temporally trim the video.  See self.commandline() for the updated filter chain to run.
        """
        assert (endframe is None or startframe <= endframe) and startframe >= 0, "Invalid start and end frames (%s, %s)" % (str(startframe), str(endframe))

        v = self.clone()
        if not v.isloaded():
            # -- Copied from super().clip() to allow for clip on clone (for indempotence)
            # -- This code copy is used to avoid super(Scene, self.clone()) which screws up class inheritance for iPython reload
            assert not v.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"            
            timestamp_in_seconds = ((v._startframe if v._startframe is not None else 0)+startframe)/float(v.framerate())
            v._update_ffmpeg_seek(timestamp_in_seconds)
            if endframe is not None:
                v._ffmpeg = v._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 before trim filter            
                v._ffmpeg = v._ffmpeg.trim(start=0, end=(endframe-startframe)/self.framerate())  # must be in seconds to allow for framerate conversion
            v._ffmpeg = v._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter            
            v._startframe = startframe if v._startframe is None else v._startframe + startframe  # for __repr__ only
            v._endframe = (v._startframe + (endframe-startframe)) if endframe is not None else v._endframe  # for __repr__ only
            # -- end copy
        else:
            endframe = endframe if endframe is not None else len(self._array)
            v._array = self._array[startframe:endframe]
            (v._startframe, v._endframe) = (0, endframe-startframe)
        v._tracks = {k:t.offset(dt=-startframe).truncate(startframe=0, endframe=(endframe-startframe) if endframe is not None else None) for (k,t) in v.tracks().items()}   # may be degenerate
        v._activities = {k:a.offset(dt=-startframe).truncate(startframe=0, endframe=(endframe-startframe) if endframe is not None else None) for (k,a) in v.activities().items()}  # may be degenerate
        return v.trackfilter(lambda t: len(t)>0).activityfilter(lambda a: len(a)>0)  # remove degenerate tracks and activities

    def crop(self, bb, zeropad=True):
        """Crop the video using the supplied box, update tracks relative to crop, video is zeropadded if box is outside frame rectangle"""
        assert isinstance(bb, vipy.geometry.BoundingBox), "Invalid input"
        bb = bb.int()
        bbc = bb.clone().imclipshape(self.width(), self.height()).int()
        #if zeropad and bb != bbc:
        #    self.zeropad(bb.width()-bbc.width(), bb.height()-bbc.height())  
        #    bb = bb.offset(bb.width()-bbc.width(), bb.height()-bbc.height())            
        super().crop(bb, zeropad=zeropad)  # range check handled here to correctly apply zeropad
        bb = bb if zeropad else bbc
        self._tracks = {k:t.offset(dx=-bb.xmin(), dy=-bb.ymin()) for (k,t) in self.tracks().items()}
        return self
    
    def zeropad(self, padwidth, padheight):
        """Zero pad the video with padwidth columns before and after, and padheight rows before and after
           Update tracks accordingly. 

        """
        
        assert isinstance(padwidth, int) and isinstance(padheight, int)
        super().zeropad(padwidth, padheight)  
        self._tracks = {k:t.offset(dx=padwidth, dy=padheight) for (k,t) in self.tracks().items()}
        return self
        
    def fliplr(self):
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.fliplr(H,W) for (k,t) in self.tracks().items()}
        super().fliplr()
        return self

    def flipud(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.flipud(H,W) for (k,t) in self.tracks().items()}
        super().flipud()
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90ccw(H,W) for (k,t) in self.tracks().items()}
        super().rot90ccw()
        return self

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        self._tracks = {k:t.rot90cw(H,W) for (k,t) in self.tracks().items()}
        super().rot90cw()
        return self

    def resize(self, rows=None, cols=None):
        """Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided"""
        assert rows is not None or cols is not None, "Invalid input"
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter, manually set this if known
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self._tracks = {k:t.scalex(sx) for (k,t) in self.tracks().items()}
        self._tracks = {k:t.scaley(sy) for (k,t) in self.tracks().items()}
        super().resize(rows=rows, cols=cols)        
        return self

    def mindim(self, dim=None):
        """Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio"""
        assert dim is None or not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return min(self.shape()) if dim is None else self.resize(cols=dim) if W<H else self.resize(rows=dim)

    def maxdim(self, dim=None):
        """Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio"""
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        return max(H,W) if dim is None else (self.resize(cols=dim) if W>H else self.resize(rows=dim))        
    
    def rescale(self, s):
        """Spatially rescale the scene by a constant scale factor.

        Args:
            s: [float] Scale factor > 0 to isotropically scale the image.
        """
        assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"                
        self._tracks = {k:t.rescale(s) for (k,t) in self.tracks().items()}
        super().rescale(s)
        return self

    def startframe(self):
        return self._startframe

    def extrapolate(self, f, dt=None):
        """Extrapolate the video to frame f and add the extrapolated tracks to the video"""
        return self.trackmap(lambda t: t.add(f, t.linear_extrapolation(f, dt=dt if dt is not None else self.framerate()), strict=False))
        
    def dedupe(self, spatial_iou_threshold=0.8, dt=5):
        """Find and delete duplicate tracks by track segmentiou() overlap.
        
        Algorithm
        - For each pair of tracks with the same category, find the larest temporal segment that contains both tracks.
        - For this segment, compute the IOU for each box interpolated at a stride of dt frames
        - Compute the mean IOU for this segment.  This is the segment IOU. 
        - If the segment IOU is greater than the threshold, merge the shorter of the two tracks with the current track.  

        """
        deleted = set([])
        for tj in sorted(self.tracklist(), key=lambda t: len(t), reverse=True):  # longest to shortest
            for (s, ti) in sorted([(0,t) if (len(tj) < len(t) or t.id() in deleted or t.id() == tj.id() or t.category() != tj.category()) else (tj.fragmentiou(t, dt=dt), t) for t in self.tracklist()], key=lambda x: x[0], reverse=True):
                if s > spatial_iou_threshold:  # best mean framewise overlap during overlapping segment of two tracks (ti, tj)
                    print('[vipy.video.dedupe]: merging duplicate track "%s" (id=%s) which overlaps with "%s" (id=%s)' % (ti, ti.id(), tj, tj.id()))
                    self.tracks()[tj.id()] = tj.union(ti)  # merge
                    self.activitymap(lambda a: a.replace(ti, tj))  # replace merged track reference in activity
                    deleted.add(ti.id())
        self.trackfilter(lambda t: t.id() not in deleted)  # remove duplicate tracks
        return self
    
    def union(self, other, temporal_iou_threshold=0.5, spatial_iou_threshold=0.6, strict=True, overlap='average', percentilecover=0.8, percentilesamples=100, activity=True, track=True):
        """Compute the union two scenes as the set of unique activities and tracks.  

           A pair of activities or tracks are non-unique if they overlap spatially and temporally by a given IoU threshold.  Merge overlapping tracks. 
           Tracks are merged by considering the mean IoU at the overlapping segment of two tracks with the same category greater than the provided spatial_iou_threshold threshold
           Activities are merged by considering the temporal IoU of the activities of the same class greater than the provided temporal_iou_threshold threshold
  
           Input:
             -Other: Scene or list of scenes for union.  Other may be a clip of self at a different framerate, spatial isotropic scake, clip offset
             -spatial_iou_threshold:  The intersection over union threshold for the mean of the two segments of an overlapping track, Disable by setting to 1.0
             -temporal_iou_threshold:  The intersection over union threshold for a temporal bounding box for a pair of activities to be declared duplicates.  Disable by setting to 1.0
             -strict:  Require both scenes to share the same underlying video filename
             -overlap=['average', 'replace', 'keep']
                -average: Merge two tracks by averaging the boxes (average=True) if overlapping
                -replace:  merge two tracks by replacing overlapping boxes with other (discard self)
                -keep: merge two tracks by keeping overlapping boxes with other (discard other)
             -percentilecover [0,1]:  When determining the assignment of two tracks, compute the percentilecover of two tracks by ranking the cover in the overlapping segment and computing the mean of the top-k assignments, where k=len(segment)*percentilecover.
             -percentilesamples [>1]:  the number of samples along the overlapping scemgne for computing percentile cover
             -activity [bool]: union() of activities only
             -track [bool]: union() of tracks only

           Output:
             -Updates this scene to include the non-overlapping activities from other.  By default, it takes the strict union of all activities and tracks. 

           Notes:
             -This is useful for merging scenes computed using a lower resolution/framerate/clipped  object or activity detector without running the detector on the high-res scene
             -This function will preserve the invariance for v == v.clear().union(v.rescale(0.5).framerate(5).activityclip()), to within the quantization error of framerate() downsampling.
             -percentileiou is a robust method of track assignment when boxes for two tracks (e.g. ground truth and detections) where one track may deform due to occlusion.
        """
        assert overlap in ['average', 'replace', 'keep'], "Invalid input - 'overlap' must be in [average, replace, keep]"
        assert spatial_iou_threshold >= 0 and spatial_iou_threshold <= 1, "invalid spatial_iou_threshold, must be between [0,1]"
        assert temporal_iou_threshold >= 0 and temporal_iou_threshold <= 1, "invalid temporal_iou_threshold, must be between [0,1]"        
        assert percentilesamples >= 1, "invalid samples, must be >= 1"
        if not activity and not track:
            return self  # nothing to do

        sc = self.clone()  # do not change self yet, make a copy then merge at the end
        for o in tolist(other):
            assert isinstance(o, Scene), "Invalid input - must be vipy.video.Scene() object and not type=%s" % str(type(o))

            if strict:
                assert sc.filename() == o.filename(), "Invalid input - Scenes must have the same underlying video.  Disable this with strict=False."
            oc = o.clone()   # do not change other, make a copy

            # Key collision?
            if len(set(sc.tracks().keys()).intersection(set(oc.tracks().keys()))) > 0:
                print('[vipy.video.union]: track key collision - Rekeying other... Use other.rekey() to suppress this warning.')
                oc.rekey()
            if len(set(sc.activities().keys()).intersection(set(oc.activities().keys()))) > 0:
                print('[vipy.video.union]: activity key collision - Rekeying other... Use other.rekey() to suppress this warning.')                
                oc.rekey()

            # Similarity transform?  Other may differ from self by a temporal scale (framerate), temporal translation (clip) or spatial isotropic scale (rescale)
            assert np.isclose(sc.aspect_ratio(), oc.aspect_ratio(), atol=1E-2), "Invalid input - Scenes must have the same aspect ratio"
            if sc.width() != oc.width():
                oc = oc.rescale(sc.width() / oc.width())   # match spatial scale
            if not np.isclose(sc.framerate(), oc.framerate(), atol=1E-3):
                oc = oc.framerate(sc.framerate())   # match temporal scale (video in oc will not match, only annotations)
            if sc.startframe() != oc.startframe():
                dt = (oc.startframe() if oc.startframe() is not None else 0) - (sc.startframe() if sc.startframe() is not None else 0)
                oc = oc.trackmap(lambda t: t.offset(dt=dt)).activitymap(lambda a: a.offset(dt=dt))  # match temporal translation of tracks and activities
            oc = oc.trackfilter(lambda t: ((not t.isdegenerate()) and len(t)>0), activitytrack=False)  

            # Merge other tracks into selfclone: one-to-many mapping from self to other
            merged = {}  # dictionary mapping trackid in other to the trackid in self, each track in other can be merged at most once
            for ti in sorted(sc.tracklist(), key=lambda t: len(t), reverse=True):  # longest to shortest
                for tj in sorted(oc.tracklist(), key=lambda t: len(t), reverse=True):  
                    if ti.category() == tj.category() and (tj.id() not in merged) and tj.segment_percentilecover(sc.track(ti.id()), percentile=percentilecover, samples=percentilesamples) > spatial_iou_threshold:  # mean framewise overlap during overlapping segment of two tracks
                        sc.tracks()[ti.id()] = sc.track(ti.id()).union(tj, overlap=overlap)  # merge duplicate/fragmented tracks from other into self, union() returns clone
                        merged[tj.id()] = ti.id()  
                        print('[vipy.video.union]: merging track "%s"(id=%s) + "%s"(id=%s) for scene "%s"' % (str(ti), str(ti.id()), str(tj), str(tj.id()), str(sc)))                        
            oc.trackfilter(lambda t: t.id() not in merged, activitytrack=False)  # remove duplicate other track for final union

            # Merge other activities into selfclone: one-to-one mapping
            for (i,j) in merged.items():  # i=id of other, j=id of self
                oc.activitymap(lambda a: a.replaceid(i, j) if a.hastrack(i) else a)  # update track IDs referenced in activities for merged tracks
            for (i,ai) in sc.activities().items():
                for (j,aj) in oc.activities().items():
                    if ai.category() == aj.category() and set(ai.trackids()) == set(aj.trackids()) and ai.temporal_iou(aj) > temporal_iou_threshold:
                        oc.activityfilter(lambda a: a.id() != j)  # remove duplicate activity from final union
            oc.activityfilter(lambda a: len(a.tracks())>0)  # remove empty activities not merged

            # Union
            sc.tracks().update(oc.tracks())
            sc.activities().update(oc.activities())

        # Final union of unique tracks/activities
        if track:
            self.tracks(sc.tracklist())  # union of tracks only
        if activity:
            self.activities(sc.activitylist())  # union of activities only: may reference tracks not in self of track=False
        return self        

    def annotate(self, outfile=None, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], mutator=None, timestamp=None, timestampcolor='black', timestampfacecolor='white', verbose=True):
        """Generate a video visualization of all annotated objects and activities in the video.
        
        The annotation video will be at the resolution and framerate of the underlying video, and pixels in this video will now contain the overlay.
        This function does not play the video, it only generates an annotation video frames.  Use show() which is equivalent to annotate().saveas().play()
        
        Args:
            outfile: [str] An optional file to stream the anntation to without storing the annotated video in memory
            fontsize: [int] The fontsize of bounding box captions, used by matplotlib
            captionoffset: (tuple) The (x,y) offset relative to the bounding box to place the caption for each box.
            textfacecolor: [str] The color of the text in the bounding box caption.  Must be in `vipy.gui.using_matplotlib.colorlist`.
            textfacealpha: [float] The transparency of the text in the bounding box caption.  Must be in [0,1], where 0=transparent and 1=opaque.
            shortlabel: [bool] If true, display the shortlabel for each object in the scene, otherwise show the full category
            boxalpha: [float]  The transparency of the box face behind the text.  Must be in [0,1], where 0=transparent and 1=opaque.
            d_category2color: [dict]  A dictionary mapping categories of objects in the scene to their box colors.  Named colors must be in `vipy.gui.using_matplotlib.colorlist`. 
            categories: [list]  Only show these categories, or show them all if None
            nocaption_withstring: [list]:  Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list
            nocaption: [bool] If true, do not show any captions, just boxes
            mutator: [lambda] A lambda function that will mutate an image to allow for complex visualizations.  This should be a mutator like `vipy.image.mutator_show_trackid`.
            timestamp: [bool] If true, show a semitransparent timestamp (when the annotation occurs, not when the video was collected) with frame number in the upper left corner of the video
            timestampcolor: [str] The color of the timstamp text.  Named colors must be in `vipy.gui.using_matplotlib.colorlist`.
            timestampfacecolor: [str]  The color of the timestamp background.  Named colors must be in `vipy.gui.using_matplotlib.colorlist`.  
            verbose: [bool] Show more helpful messages if true

        Returns:
            A `vipy.video.Video` with annotations in the pixels.  If outfile is provided, then the returned video will be flushed.  

        .. note::  In general, this function should not be run on very long videos without the outfile kwarg, as it requires loading the video framewise into memory.  
        """
        if verbose:
            print('[vipy.video.annotate]: Annotating video ...')  
            
        f_mutator = mutator if mutator is not None else vipy.image.mutator_show_jointlabel()
        f_timestamp = (lambda k: '%s %d' % (vipy.util.clockstamp(), k)) if timestamp is True else timestamp

        if outfile is None:        
            assert self.load().isloaded(), "Load() failed"
            imgs = [f_mutator(self[k].clone(), k).savefig(fontsize=fontsize,
                                                  captionoffset=captionoffset,
                                                  textfacecolor=textfacecolor,
                                                  textfacealpha=textfacealpha,
                                                  shortlabel=shortlabel,
                                                  boxalpha=boxalpha,
                                                  d_category2color=d_category2color,
                                                  categories=categories,
                                                  nocaption=nocaption,
                                                  timestampcolor=timestampcolor,
                                                  timestampfacecolor=timestampfacecolor,
                                                  timestamp=f_timestamp(k) if timestamp is not None else None,
                                                  figure=1 if k<(len(self)-1) else None,  # cleanup on last frame
                                                  nocaption_withstring=nocaption_withstring).numpy() for k in range(0, len(self))]
            
            # Replace pixels with annotated pixels and downcast object to vipy.video.Video (since there are no more objects to show)
            return vipy.video.Video(array=np.stack([np.array(PIL.Image.fromarray(img).convert('RGB')) for img in imgs], axis=0), framerate=self.framerate(), attributes=self.attributes)  # slow for large videos
        else:
            # Stream to output video without loading all frames into memory
            n = self.duration_in_frames_of_videofile() if not self.isloaded() else len(self)
            vo = vipy.video.Video(filename=outfile, framerate=self.framerate())
            with vo.stream(overwrite=True) as so:
                for (k,im) in enumerate(self.stream()):
                    so.write(f_mutator(im.clone(), k).savefig(fontsize=fontsize,
                                                      captionoffset=captionoffset,
                                                      textfacecolor=textfacecolor,
                                                      textfacealpha=textfacealpha,
                                                      shortlabel=shortlabel,
                                                      boxalpha=boxalpha,
                                                      d_category2color=d_category2color,
                                                      categories=categories,
                                                      nocaption=nocaption,
                                                      timestampcolor=timestampcolor,
                                                      timestampfacecolor=timestampfacecolor,
                                                      timestamp=f_timestamp(k) if timestamp is not None else None,
                                                      figure=1 if k<(n-1) else None,  # cleanup on last frame
                                                      nocaption_withstring=nocaption_withstring).rgb())
            return vo


    def _show(self, outfile=None, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], notebook=False, timestamp=None, timestampcolor='black', timestampfacecolor='white'):
        """Generate an annotation video saved to outfile (or tempfile if outfile=None) and show it using ffplay when it is done exporting.  Do not modify the original video buffer.  Returns a clone of the video with the shown annotation."""
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
                                     timestampcolor=timestampcolor,
                                     timestampfacecolor=timestampfacecolor,
                                     timestamp=timestamp,
                                     nocaption_withstring=nocaption_withstring).saveas(outfile).play(notebook=notebook)
    

    def show(self, outfile=None, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], figure=1, fps=None, timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None):
        """Faster show using interative image show for annotated videos.  This can visualize videos before video rendering is complete, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates."""
        fps = min(fps, self.framerate()) if fps is not None else self.framerate()
        assert fps > 0, "Invalid display framerate"
        f_timestamp = (lambda k: '%s %d' % (vipy.util.clockstamp(), k)) if timestamp is True else timestamp
        f_mutator = mutator if mutator is not None else vipy.image.mutator_show_jointlabel()        
        if not self.isdownloaded() and self.hasurl():
            self.download()
        with Stopwatch() as sw:            
            for (k,im) in enumerate(self.load() if self.isloaded() else self.stream()):
                time.sleep(max(0, (1.0/self.framerate())*int(np.ceil((self.framerate()/fps))) - sw.since()))                
                f_mutator(im,k).show(categories=categories,
                                     figure=figure,
                                     nocaption=nocaption,
                                     nocaption_withstring=nocaption_withstring,
                                     fontsize=fontsize,
                                     boxalpha=boxalpha,
                                     d_category2color=d_category2color,
                                     captionoffset=captionoffset,
                                     textfacecolor=textfacecolor,
                                     textfacealpha=textfacealpha,
                                     timestampcolor=timestampcolor,
                                     timestampfacecolor=timestampfacecolor,
                                     timestamp=f_timestamp(k) if timestamp is not None else None,
                                     shortlabel=shortlabel)
                
                if vipy.globals._user_hit_escape():
                    break
        vipy.show.close(figure)
        return self

    def thumbnail(self, outfile=None, frame=0, fontsize=10, nocaption=False, boxalpha=0.25, dpi=200, textfacecolor='white', textfacealpha=1.0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile if provided, otherwise return vipy.image.Scene"""
        im = self.frame(frame, img=self.preview(framenum=frame).array())
        return im.savefig(outfile=outfile, fontsize=fontsize, nocaption=nocaption, boxalpha=boxalpha, dpi=dpi, textfacecolor=textfacecolor, textfacealpha=textfacealpha) if outfile is not None else im
    
    def stabilize(self, flowdim=256):
        """Background stablization using flow based stabilization masking foreground region.  This will output a video with all frames aligned to the first frame, such that the background is static."""
        from vipy.flow import Flow  # requires opencv
        return Flow(flowdim=flowdim).stabilize(self.clone(), residual=True)
    
    def pixelmask(self, pixelsize=8):
        """Replace all pixels in foreground boxes with pixelation"""
        for im in self: 
            im.pixelmask(pixelsize)  # shared numpy array
        return self

    def binarymask(self):
        """Replace all pixels in foreground boxes with white, zero in background"""        
        for im in self:
            im.binarymask()  # shared numpy array
        return self

    def asfloatmask(self, fg=1.0, bg=0.0):
        """Replace all pixels in foreground boxes with fg, and bg in background, return a copy"""
        array = np.full( (len(self.load()), self.height(), self.width(), 1), dtype=np.float32, fill_value=bg)
        for (k,im) in enumerate(self):
            for bb in im.objects():
                if bb.hasintersection(im.imagebox()):
                    array[k, int(round(bb._ymin)):int(round(bb._ymax)), int(round(bb._xmin)):int(round(bb._xmax))] = fg   # does not need imclip
        return vipy.video.Video(array=array, framerate=self.framerate(), colorspace='float')

    
    def meanmask(self):
        """Replace all pixels in foreground boxes with mean color"""        
        for im in self:
            im.meanmask()  # shared numpy array
        return self

    def fgmask(self):
        """Replace all pixels in foreground boxes with zero"""        
        for im in self:
            im.fgmask()  # shared numpy array
        return self

    def zeromask(self):
        """Alias for fgmask"""
        return self.fgmask()
    
    def blurmask(self, radius=7):
        """Replace all pixels in foreground boxes with gaussian blurred foreground"""        
        for im in self:
            im.blurmask(radius)  # shared numpy array
        return self

    def downcast(self):
        """Cast the object to a `vipy.video.Video` class"""
        self.__class__ = vipy.video.Video
        return self

    def merge_tracks(self, dilate_height=2.0, dilate_width=2.0, framedist=5):
        """Merge tracks if a track endpoint dilated by a fraction overlaps exactly one track startpoint, and the endpoint and startpoint are close enough together temporally.
        
        .. note::
        - This is useful for continuing tracking when the detection framerate was too low and the assignment falls outside the measurement gate.
        - This will not work for complex scenes, as it assumes that there is exactly one possible continuation for a track.  
        
        """
        merged = set([])
        for ti in sorted(self.tracklist(), key=lambda t: t.startframe()):
            for tj in sorted(self.tracklist(), key=lambda t: t.startframe()):
                if (tj.id() not in merged) and (ti.id() != tj.id()) and (tj.startframe() >= ti.endframe()) and ((tj.startframe()-ti.endframe()) <= framedist) and (ti.category() == tj.category()):
                    di = ti[ti.endframe()].dilate_height(dilate_height).dilate_width(dilate_width)
                    dj = tj[tj.startframe()]
                    if di.iou(dj) > 0 and not any([di.iou(tk[tj.startframe()]) > 0 for tk in self.tracklist() if (tk.id() not in [ti.id(), tj.id()]) and tk.during(tj.startframe())]):
                        self.tracks()[ti.id()] = ti.union(tj)  # Merge tracks that are within gating distance
                        self.delete(tj.id())  # remove merged track
                        merged.add(tj.id())
                        break
        return self

    def assign(self, frame, dets, minconf=0.2, maxhistory=5, activityiou=0.5, trackcover=0.2, trackconfsamples=4, gate=0, activitymerge=True, activitynms=False):
        """Assign a list of vipy.object.Detections at frame k to scene by greedy track association. In-place update.
        
        Args:
            miniou: [float] the minimum temporal IOU for activity assignment
            minconf: [float] the minimum confidence for a detection to be considered as a new track
            maxhistory: [int]  the maximum propagation length of a track with no measurements, the frame history ised for velocity estimates  
            trackconfsamples: [int]  the number of uniformly spaced samples along a track to compute a track confidence
            gate: [int] the gating distance in pixels used for assignment of fast moving detections.  Useful for low detection framerates if a detection does not overlap with the track.
            trackcover: [float] the minimum cover necessary for assignment of a detection to a track
            activitymerge: [bool] if true, then merge overlapping activity detections of the same track and category, otherwise each activity detection is added as a new detection
            activitynms: [bool] if true, then perform non-maximum suppression of activity detections of the same actor and category that overlap more than activityiou

        Returns:
            This video object with each det assigned to correpsonding track or activity.

        """
        assert dets is None or all([isinstance(d, vipy.object.Detection) or isinstance(d, vipy.activity.Activity) for d in tolist(dets)]), "invalid input"
        assert frame >= 0 and minconf >= 0 and minconf <= 1.0 and maxhistory > 0, "invalid input"
        
        if dets is None or len(tolist(dets)) == 0:
            return self
        dets = tolist(dets)

        if any([d.confidence() is None for d in dets]):
            warnings.warn('Removing %d detections with no confidence' % len([d.confidence() is None for d in dets]))
            dets = [d for d in dets if d.confidence() is not None]
        objdets = [d for d in dets if isinstance(d, vipy.object.Detection)]
        activitydets = [d for d in dets if isinstance(d, vipy.activity.Activity)]        

        # Object detection to track assignment
        if len(objdets) > 0:
            # Track propagation:  Constant velocity motion model for active tracks 
            t_ref = [(t, t.linear_extrapolation(frame, dt=maxhistory, shape=False)) for (k,t) in self.tracks().items() if ((frame - t.endframe()) <= maxhistory)]
            trackarea = [ti.area() for (t,ti) in t_ref]
            detarea = [d.area() for d in objdets]
            
            # Track assignment:
            #   - Each track is assigned at most one detection
            #   - Each detection is assigned to at most one track.  
            #   - Assignment is the highest confidence maximum overlapping detection within tracking gate
            trackconf = {t.id():t.confidence(samples=trackconfsamples) for (t, ti) in t_ref}
            assignments = [(t, d.confidence(), d.iou(ti, area=detarea[j], otherarea=trackarea[i]), d.shapeiou(ti, area=detarea[j], otherarea=trackarea[i]), d.maxcover(ti, area=detarea[j], otherarea=trackarea[i]), d)
                           for (i, (t, ti)) in enumerate(t_ref)
                           for (j,d) in enumerate(objdets)
                           if (t.category() == d.category() and
                               (((ti._xmax if ti._xmax < d._xmax else d._xmax) - (ti._xmin if ti._xmin > d._xmin else d._xmin)) > 0 and
                                ((ti._ymax if ti._ymax < d._ymax else d._ymax) - (ti._ymin if ti._ymin > d._ymin else d._ymin)) > 0))]
            
            assigned = set([])        
            posconf = min([d.confidence() for d in objdets]) if len(objdets)>0 else 0
            assignments.sort(key=lambda x: (x[1]+posconf)*(x[2]+x[3]+x[4])+trackconf[x[0].id()], reverse=True)  # in-place
            for (t, conf, iou, shapeiou, cover, d) in assignments:
                if cover > (trackcover if len(t)>1 else 0):  # the highest confidence detection within the iou gate (or any overlap if not yet enough history for velocity estimate) 
                    if (t.id() not in assigned and d.id() not in assigned):  # not assigned yet, assign it!
                        self.track(t.id()).update(frame, d.clone())  # track assignment! (clone required)
                        assigned.add(t.id())  # cannot assign again to this track
                        assigned.add(d.id())  # mark detection as assigned
                
            # Track spawn from unassigned and unexplained detections 
            for (j,d) in enumerate(objdets):                
                if (d.id() not in assigned):
                    if (d.confidence() >= minconf and not any([t.linear_extrapolation(frame, dt=maxhistory, shape=False).maxcover(d, otherarea=detarea[j]) >= 0.7 for (i,(t,ti)) in enumerate(t_ref) if t.category() == d.category()])):
                        gated = [(t, t.linear_extrapolation(frame, dt=maxhistory, shape=False)) for (t,ti) in t_ref if (t.id() not in assigned and t.category() == d.category())] if gate>0 else []
                        gated = sorted([(t, ti) for (t, ti) in gated if ti.hasintersection(d, gate=gate)], key=lambda x: d.sqdist(x[1]))
                        if len(gated) > 0:
                            self.track(gated[0][0].id()).update(frame, d.clone())  # track assignment! (clone required)
                            assigned.add(gated[0][0].id())
                            assigned.add(d.id())
                        else:
                            assigned.add(self.add(vipy.object.Track(keyframes=[frame], boxes=[d.clone()], category=d.category(), framerate=self.framerate()), rangecheck=False))  # clone required
                            assigned.add(d.id())

        # Activity assignment
        if len(activitydets) > 0:
            assert all([d.actorid() in self.tracks() for d in activitydets]), "Invalid activity"
            assigned = set([])
            if activitymerge:
                minframe = min([a._startframe for a in activitydets]) 
                activities = [a for a in self.activities().values() if a._endframe >= minframe]
                for d in activitydets:
                    for a in activities:
                        if (a._label == d._label) and (a._actorid == d._actorid) and a.hasoverlap(d, activityiou): 
                            a.union(d)  # activity assignment 
                            assigned.add(d._id)
                            break  # assigned, early exit
                        
            if activitynms:
                minframe = min([a._startframe for a in activitydets]) 
                activities = sorted([a for a in self.activities().values() if a._endframe >= minframe], key=lambda a: a.confidence(), reverse=True)
                for d in sorted(activitydets, key=lambda x: x.confidence(), reverse=True):
                    for a in activities:
                        if (a._label == d._label) and (a._actorid == d._actorid) and a.hasoverlap(d, activityiou):
                            assigned.add(a._id if d.confidence()>a.confidence() else d._id)  # suppressed
                for id in assigned:
                    if id in self._activities:
                        del self._activities[id]  # suppression, faster than self.activityfilter(lambda a: a.id() in assigned)
                                    
            # Activity construction from unassigned detections
            for d in activitydets:
                if d._id not in assigned:
                    self.add(d.clone())  

        return self

    
    
def RandomVideo(rows=None, cols=None, frames=None):
    """Return a random loaded vipy.video.video.
    
    Useful for unit testing, minimum size (32x32x32) for ffmpeg
    """
    rows = np.random.randint(256, 1024) if rows is None else rows
    cols = np.random.randint(256, 1024) if cols is None else cols
    frames = np.random.randint(32, 256) if frames is None else frames
    assert rows>32 and cols>32 and frames>=32    
    return Video(array=np.uint8(255 * np.random.rand(frames, rows, cols, 3)), colorspace='rgb')


def RandomScene(rows=None, cols=None, frames=None):
    """Return a random loaded vipy.video.Scene.
    
    Useful for unit testing.
    """
    v = RandomVideo(rows, cols, frames)
    (rows, cols) = v.shape()
    tracks = [vipy.object.Track(label='track%d' % k, shortlabel='t%d' % k,
                                keyframes=[0, np.random.randint(50,100), 150],
                                framerate=30, 
                                boxes=[vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,32)]

    activities = [vipy.activity.Activity(label='activity%d' % k, shortlabel='a%d' % k, tracks=[tracks[j].id() for j in [np.random.randint(32)]], startframe=np.random.randint(50,99), endframe=np.random.randint(100,150), framerate=30) for k in range(0,32)]   
    return Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities, framerate=30)


def RandomSceneActivity(rows=None, cols=None, frames=256):
    """Return a random loaded vipy.video.Scene.

    Useful for unit testing.
    """    
    v = RandomVideo(rows, cols, frames)
    (rows, cols) = v.shape()
    tracks = [vipy.object.Track(label=['Person','Vehicle','Object'][k], shortlabel='track%d' % k, boundary='strict', 
                                keyframes=[0, np.random.randint(50,100), np.random.randint(50,150)],
                                framerate=30,
                                boxes=[vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,3)]

    activities = [vipy.activity.Activity(label='Person Carrying', shortlabel='Carry', tracks=[tracks[0].id(), tracks[1].id()], startframe=np.random.randint(20,50), endframe=np.random.randint(70,100), framerate=30)]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities, framerate=30)

    return ims
    
def EmptyScene():
    """Return an empty scene""" 
    return vipy.video.Scene(array=np.zeros((1,1,1,3), dtype=np.uint8))
