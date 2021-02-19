import os
import sys
import dill
from vipy.globals import print
from vipy.util import remkdir, tempMP4, isurl, \
    isvideourl, templike, tempjpg, filetail, tempdir, isyoutubeurl, try_import, isnumpy, temppng, \
    istuple, islist, isnumber, tolist, filefull, fileext, isS3url, totempdir, flatlist, tocache, premkdir, writecsv, iswebp, ispng, isgif, filepath, Stopwatch, toextension, isjsonfile
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

    >>> vid = vipy.video.Video(array=frames, colorspace='rgb')
    
    The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video
    Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load().

    """
    def __init__(self, filename=None, url=None, framerate=30.0, attributes=None, array=None, colorspace=None, startframe=None, endframe=None, startsec=None, endsec=None, frames=None, probeshape=False):
        self._url = None
        self._filename = None
        self._array = None
        self._colorspace = None
        self._ffmpeg = None
        self._framerate = framerate

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
        (self._startsec, self._endsec) = (None, None)      # __repr__ only  

        # Input filenames
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
            self._url = url
        if filename is not None:
            self._filename = os.path.normpath(os.path.expanduser(filename))
        elif self._url is not None:
            if isS3url(self._url):
                self._filename = totempdir(self._url)  # Preserve S3 Object ID
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            elif isyoutubeurl(self._url):
                self._filename = os.path.join(tempdir(), '%s' % self._url.split('?')[1].split('&')[0])
            else:
                self._filename = totempdir(self._url)  
            if vipy.globals.cache() is not None and self._filename is not None:
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
        if startsec is not None and endsec is not None:
            (self._startsec, self._endsec) = (startsec, endsec)            
            self.cliptime(startsec, endsec)
            
        # Array input
        assert not (array is not None and frames is not None)
        if array is not None:
            self.array(array)
            self.colorspace(colorspace)
        elif frames is not None:
            self.fromframes(frames)

            
        
    @classmethod
    def cast(cls, v):
        assert isinstance(v, vipy.video.Video), "Invalid input - must be derived from vipy.video.Video"
        v.__class__ = vipy.video.Video
        return v
            
    @classmethod
    def from_json(cls, s):
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
        """Number of frames in the video if loaded, else zero.  Do not automatically trigger a load, since this can interact in unexpected ways with other tools that depend on fast __len__()"""
        if not self.isloaded():
            warnings.warn('Load() video to see number of frames - Returning zero')  # should this just throw an exception?
        return len(self.array()) if self.isloaded() else 0

    def __getitem__(self, k):
        """Return the kth frame as an vipy.image object"""
        return self.frame(k)

    def metadata(self):
        """Return a dictionary of metadata about this video"""
        return self.attributes

    def videoid(self):
        """Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by hashing the filename() and url().

           Notes:
             - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change.
             - If a video does not have a filename or URL or a video ID in the attributes, then this will return None
             - To preserve a video ID independent of transformations, set self.setattribute('video_id', $MY_ID)
        """
        return self.attributes['video_id'] if 'video_id' in self.attributes else hashlib.sha1(str(str(self.filename())+str(self.url())).encode("UTF-8")).hexdigest() if (self.filename() is not None or self.url() is not None) else None
    
    def frame(self, k, img=None):
        """Alias for self.__getitem__[k]"""
        assert isinstance(k, int) and k>=0, "Frame index must be non-negative integer"
        return Image(array=img if img is not None else (self._array[k] if self.isloaded() else self.preview(k).array()), colorspace=self.colorspace())       
        
    def __iter__(self):
        """Iterate over loaded video, yielding mutable frames"""
        self.load().numpy()  # triggers load and copy of read-only video buffer, mutable
        with np.nditer(self._array, op_flags=['readwrite']) as it:
            for k in range(0, len(self)):
                self._currentframe = k    # used only for incremental add()                
                yield self.__getitem__(k)
            self._currentframe = None    # used only for incremental add()

    def store(self):
        """Store the current video file as an attribute of this object.  Useful for archiving an object to be fully self contained without any external references.  
        
           -Remove this stored video using unstore()
           -Unpack this stored video and set up the video chains using restore() 
           -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string.
           -Useful for creating a single self contained object for distributed processing.  

           >>> v == v.store().restore(v.filename()) 

        """
        assert self.hasfilename(), "Video file not found"
        with open(self.filename(), 'rb') as f:
            self.attributes['__video__'] = f.read()
        return self

    def unstore(self):
        """Delete the currently stored video from store()"""
        return self.delattribute('__video__')

    def restore(self, filename):
        """Save the currently stored video to filename, and set up filename"""
        assert self.hasattribute('__video__'), "Video not stored"
        with open(filename, 'wb') as f:
            f.write(self.attributes['__video__'])
        return self.filename(filename)                
        
    def stream(self, write=False, overwrite=False):
        """Iterator to yield frames streaming from video
        
           * Using this iterator may affect PDB debugging due to stdout/stdin redirection.  Use ipdb instead.
           * FFMPEG stdout pipe may screw up bash shell newlines, requiring issuing command "reset"  

        """

        class Stream(object):
            def __init__(self, v, bufsize=1024):
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
                    self._write_pipe = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=self._framerate) \
                                             .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2') \
                                             .output(filename=self._outfile, pix_fmt='yuv420p', vcodec=self._vcodec) \
                                             .overwrite_output() \
                                             .global_args('-cpuflags', '0', '-loglevel', 'quiet' if not vipy.globals.isdebug() else 'debug') \
                                             .run_async(pipe_stdin=True)
                return self
            
            def __exit__(self, type, value, tb):
                """Write pipe context manager"""
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
                p = self._video._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24').global_args('-nostdin', '-loglevel', 'debug' if vipy.globals.isdebug() else 'quiet').run_async(pipe_stdout=True, pipe_stderr=True)
                assert p is not None, "Invalid read pipe"
                return p
            
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
                                if pipe.returncode != 0:
                                    raise ValueError('Stream iterator failed with error "%s"' % str(pipe.stderr.readlines()))
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
                return self._video.thumbnail(frame=k)  # this is inefficient

            def write(self, im, flush=False):
                """Write individual frames to write stream"""
                if self._shape is None:
                    self._shape = im.shape()
                    assert im.channels() == 3, "RGB frames required"
                    self.__enter__()
                assert self._write_pipe is not None, "Write stream cannot be initializedy"                
                assert im.shape() == self._shape, "Shape cannot change during writing"
                self._write_pipe.stdin.write(im.array().astype(np.uint8).tobytes())
                if flush:
                    self._write_pipe.stdin.flush()  # do we need this?
                
            def clip(self, n, m=1, continuous=False, tracks=True, activities=True):
                """Stream clips of length n such that the yielded video clip contains frame(0) to frame(-n), and next contains frame(m) to frame (-(n+m)). 
                
                   - n [int]: the length of the clip
                   - m [int]: the stride between clips
                   - continuous [bool]:  if true, then yield None for the frames between the strides so that a clip is yielded on every frame
                   - activities [bool]:  if false, then activities from the source video are not copied into the clip
                   - tracks [bool]:  if false, then tracks from the source video are not copied into the clip
                """
                assert isinstance(n, int) and n>0, "Clip length must be a positive integer"
                assert isinstance(m, int) and m>0, "Clip stride must be a positive integer"                
                
                def _f_threadloop(pipe, queue, height, width, video, n, m, continuous, event):
                    assert pipe is not None, "invalid pipe"
                    assert queue is not None, "invalid queue"
                    frameindex = 0
                    frames = []

                    while True:
                        in_bytes = pipe.stdout.read(height * width * 3)
                        if not in_bytes:
                            queue.put( (None, None) )
                            pipe.poll()
                            if pipe.returncode != 0:
                                raise ValueError('Clip stream iterator failed with error "%s"' % str(pipe.stderr.readlines()))
                            event.wait()
                            break
                        else:
                            frames.append(np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))

                        if len(frames) > n:
                            frames.pop(0) 
                        if (frameindex-1) % m == 0 and len(frames) >= n:
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
                        yield ((vc.activities([a.clone().offset(-k+n-1).truncate(0,n) for (ak,a) in self._video.activities().items() if a.during_interval(k-n+1, k, inclusive=True)] if activities else []) 
                               .tracks([t.clone(k-n+1, k).offset(-k+n-1).truncate(0,n) for (tk,t) in self._video.tracks().items() if t.during_interval(k-n+1, k)] if tracks else []))
                               if (vc is not None and isinstance(vc, vipy.video.Scene)) else vc)
                    else:
                        e.set()
                        break
                    
            def batch(self, n):
                """Stream batches of length n such that each batch contains frames [0,n], [n+1, 2n], ...  Last batch will be ragged"""
                assert isinstance(n, int) and n>0, "batch length must be a positive integer"

                def _f_threadloop(pipe, queue, height, width, video, n, event):
                    assert pipe is not None, "invalid pipe"
                    assert queue is not None, "invalid queue"
                    frameindex = 0
                    frames = []

                    while True:
                        in_bytes = pipe.stdout.read(height * width * 3)
                        if not in_bytes:
                            queue.put((frameindex, video.clear().clone(shallow=True).array(np.stack(frames))))
                            queue.put((None, None))
                            pipe.poll()
                            if pipe.returncode != 0:
                                raise ValueError('Batch stream iterator failed with error "%s"' % str(pipe.stderr.readlines()))                                                                
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
                """Stream individual frames of video with negative offset n to the stream head. If n=-30, this full return a frame 30 frames ago"""
                assert isinstance(n, int) and n<=0, "Frame offset must be non-positive integer"
                frames = []
                for (k,im) in enumerate(self):
                    frames.append(im)                    
                    imout = frames[0]
                    frames.pop(0) if len(frames) == abs(n) else None
                    yield imout
                                          
        return Stream(self)  # do not clone


    def clear(self):
        """no-op for Video()"""
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

    def _from_ffmpeg_commandline(self, cmd):
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
            self._startframe = int(round(timestamp_in_seconds*self.framerate()))  # necessary for clip()
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
                    f = f.filter(filtername, *a, **kw)
        assert self._ffmpeg_commandline(f.output('dummyfile')) == cmd, "FFMPEG command line '%s' != '%s'" % (self._ffmpeg_commandline(f.output('dummyfile')), cmd)
        return f

    def isdirty(self):
        """Has the FFMPEG filter chain been modified from the default?  If so, then ffplay() on the video file will be different from self.load().play()"""
        return '-filter_complex' in self._ffmpeg_commandline()

    def probeshape(self):
        """Return the (height, width) of underlying video file as determined from ffprobe, this does not take into account any applied ffmpeg filters"""
        p = self.probe()
        assert len(p['streams']) > 0
        return (p['streams'][0]['height'], p['streams'][0]['width'])
        
    def duration_in_seconds_of_videofile(self):
        """Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe.  Fetch once and cache"""
        filehash = hashlib.md5(str(self.filename()).encode()).hexdigest()            
        if self.hasattribute('_duration_in_seconds_of_videofile') and self.attributes['_duration_in_seconds_of_videofile']['filehash'] == filehash:
            return self.attributes['_duration_in_seconds_of_videofile']['duration']
        else:
            d = float(self.probe()['format']['duration'])
            self.attributes['_duration_in_seconds_of_videofile'] = {'duration':d, 'filehash':filehash}  # for next time
            return d

    def duration_in_frames_of_videofile(self):
        """Return video duration of the source filename (NOT the filter chain) in frames, requires ffprobe"""
        return int(round(self.duration_in_seconds_of_videofile()*self.framerate()))
    
    def probe(self):
        """Run ffprobe on the filename and return the result as a JSON file"""
        if not has_ffprobe:
            raise ValueError('"ffprobe" executable not found on path, this is optional for vipy.video - Install from http://ffmpeg.org/download.html')            
        assert self.hasfilename(), "Invalid video file '%s' for ffprobe" % self.filename() 
        return ffmpeg.probe(self.filename())

    def print(self, prefix='', verbose=True, sleep=None):
        """Print the representation of the video - useful for debugging in long fluent chains.  Sleep is useful for adding in a delay for distributed processing"""
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
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(s=None, encode=False)

    def json(self, encode=True):
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
        """Return n frames from the clip uniformly spaced as numpy array"""
        assert self.isloaded(), "Load() is required before take()"""
        dt = int(np.round(len(self._array) / float(n)))  # stride
        return self._array[::dt][0:n]

    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations
        
           * NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()
        
        """
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

    def canload(self):
        """Return True if the video can be loaded successfully, useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version"""
        if not self.isloaded():
            try:
                self.preview()  # try to preview
                return True
            except:
                return False
        else:
            return True

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
            if copy:
                self._array.setflags(write=True)  # mutable iterators, triggers copy
            self.colorspace(None)  # must be set with colorspace() after array() before _convert()
            return self
        else:
            raise ValueError('Invalid input - array() must be numpy array')            

    def fromarray(self, array):
        """Alias for self.array(..., copy=True), which forces the new array to be a copy"""
        return self.array(array, copy=True)

    def fromframes(self, framelist, copy=True):
        """Create a video from a list of frames"""
        assert all([isinstance(im, vipy.image.Image) for im in framelist]), "Invalid input"
        return self.array(np.stack([im.array() if im.array().ndim == 3 else np.expand_dims(im.array(), 2) for im in framelist]), copy=copy).colorspace(framelist[0].colorspace())
    
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
            assert self.hasfilename(), "File not found for copy"
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
                vipy.videosearch.download(self._url, filefull(self._filename), writeurlfile=False, skip=ignoreErrors, verbose=verbose)
                for ext in ['mkv', 'mp4', 'webm']:
                    f = '%s.%s' % (self.filename(), ext)
                    if os.path.exists(f):
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
            raise ValueError('[vipy.video.load]: Video preview failed with error "%s"\n\nVideo: "%s"\n\nFFMPEG command: "%s"\n\nTry manually running this ffmpeg command to see errors.  This error usually means that the video is corrupted.' % (str(e), str(self), str(self._ffmpeg_commandline(f_prepipe.output('preview.jpg', vframes=1)))))

        # [EXCEPTION]:  UnidentifiedImageError: cannot identify image file
        #   -This may occur when the framerate of the video from ffprobe (tbr) does not match that passed to fps filter, resulting in a zero length image preview piped to stdout
        #   -This may occur after calling clip() with too short a duration, try increasing the clip to be > 1 sec
        #   -This may occur if requesting a frame number greater than the length of the video.  At this point, we do not know the video length, and cannot fail gracefully
        try:
            return Image(array=np.array(PIL.Image.open(BytesIO(out))))
        except Exception as e:
            print('[vipy.video.Video.preview][ERROR]:  %s' % str(e))
            print('   - This may occur when the framerate of the video from ffprobe (tbr) does not match that passed to fps filter, resulting in a zero length image preview piped to stdout')
            print('   - This may occur after calling clip() with too short a duration, try increasing the clip to be > 1 sec')
            print('   - This may occur if requesting a frame number greater than the length of the video.  At this point, we do not know the video length, and cannot fail gracefully')
            print('   - This may occur if the filter chain fails for some unknown reason on this video.  Try running the ffmpeg command manually for this video and looking at the FFMPEG console output (e.g. the output of v._ffmpeg_commandline())')
            print('   - %s' % (str(self)))
            raise

    def thumbnail(self, outfile=None, frame=0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile"""
        im = self.frame(frame, img=self.preview(frame).array())
        return im.savefig(outfile) if outfile is not None else im
    
    def load(self, verbose=False, ignoreErrors=False, shape=None):
        """Load a video using ffmpeg, applying the requested filter chain.  
           
             - If verbose=True. then ffmpeg console output will be displayed. 
             - If ignoreErrors=True, then all load errors are warned and skipped.  Be sure to call isloaded() to confirm loading was successful.
             - shape tuple(height, width, channels):  If provided, use this shape for reading and reshaping the byte stream from ffmpeg
             - knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size
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
            f = self._ffmpeg.output('pipe:', format='rawvideo', pix_fmt='rgb24')\
                            .global_args('-cpuflags', '0', '-loglevel', 'debug' if vipy.globals.isdebug() else 'quiet')
            (out, err) = f.run(capture_stdout=True, capture_stderr=True)
        except Exception as e:
            if not ignoreErrors:
                raise ValueError('[vipy.video.load]: Load failed with error "%s"\n\nVideo: "%s"\n\nFFMPEG command: "%s"\n\n Try setting vipy.globals.debug() to see verbose FFMPEG debugging output and rerunning or manually running the ffmpeg command line to see errors. This error usually means that the video is corrupted or that you need to upgrade your FFMPEG distribution to the latest stable version.' % (str(e), str(self), str(self._ffmpeg_commandline(f))))
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

            if not is_loadable():
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
                self._ffmpeg = self._ffmpeg.trim(start_frame=0, end_frame=(endframe-startframe))
            self._ffmpeg = self._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter            
            self._startframe = startframe if self._startframe is None else self._startframe + startframe  # for __repr__ only
            self._endframe = (self._startframe + (endframe-startframe)) if endframe is not None else endframe  # for __repr__ only
        else:
            endframe = endframe if endframe is not None else len(self._array)
            self._array = self._array[startframe:endframe]
            (self._startframe, self._endframe) = (0, endframe-startframe)
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
                            
        if (rows is None and cols is None) or (self.shape() == newshape):
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

    def maxmatte(self):
        return self.zeropad(max(1, int((max(self.shape()) - self.width())/2)), max(int((max(self.shape()) - self.height())/2), 1))
    
    def zeropad(self, padwidth, padheight):
        """Zero pad the video with padwidth columns before and after, and padheight rows before and after
           
           * NOTE: Older FFMPEG implementations can throw the error "Input area #:#:#:# not within the padded area #:#:#:# or zero-sized, this is often caused by odd sized padding. 
             Recommend calling self.cropeven().zeropad(...) to avoid this

        """
        assert isinstance(padwidth, int) and isinstance(padheight, int)        
        if not self.isloaded():
            self.shape(shape=(self.height()+2*padheight, self.width()+2*padwidth))  # manually set shape to avoid preview            
            self._ffmpeg = self._ffmpeg.filter('pad', 'iw+%d' % (2*padwidth), 'ih+%d' % (2*padheight), '%d'%padwidth, '%d'%padheight)
        elif padwidth > 0 or padheight > 0:
            self.array( np.pad(self.array(), ((0,0), (padheight,padheight), (padwidth,padwidth), (0,0)), mode='constant'), copy=False)  # this is very expensive, since np.pad() must copy (once in np.pad >=1.17)            
        return self

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
                bb = bb.offset(int(np.ceil(bb.width()-bbc.width())), int(np.ceil(bb.height()-bbc.height())))   # Shift boundingbox by padding            
            self._ffmpeg = self._ffmpeg.filter('crop', '%d' % bb.width(), '%d' % bb.height(), '%d' % bb.xmin(), '%d' % bb.ymin(), 0, 1)  # keep_aspect=False, exact=True        
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
        
           -strict=[bool]: assert that the filename must have an .webp extension
           -pause=[int]: seconds to pause between loops of the animation
           -smallest=[bool]:  create the smallest possible file but takes much longer to run
           -smaller=[bool]:  create a smaller file, which takes a little longer to run 
        """
        assert strict is False or iswebp(outfile)
        outfile = os.path.normpath(os.path.abspath(os.path.expanduser(outfile)))
        self.load().frame(0).pil().save(outfile, loop=0, save_all=True, method=6 if smallest else 3 if smaller else 0,
                                        append_images=[self.frame(k).pil() for k in range(1, len(self))],
                                        duration=[int(1000.0/self._framerate) for k in range(0, len(self)-1)] + [pause*1000])
        return outfile

    def gif(self, outfile, pause=3, smallest=False, smaller=False):
        """Save a video to an animated GIF file, with pause=N seconds between loops.  

           WARNING:  this will be very large for big videos, consider using webp instead.
           -pause=[int]: seconds to pause between loops of the animation
           -smallest=[bool]:  create the smallest possible file but takes much longer to run
           -smaller=[bool]:  create a smaller file, which takes a little longer to run 
        """        
        assert isgif(outfile)
        return self.webp(outfile, pause, strict=False, smallest=smallest, smaller=True)
        
    def saveas(self, outfile=None, framerate=None, vcodec='libx264', verbose=False, ignoreErrors=False, flush=False, pause=5):
        """Save video to new output video file.  This function does not draw boxes, it saves pixels to a new video file.

           * outfile: the absolute path to the output video file.  This extension can be .mp4 (for video) or [".webp",".gif"]  (for animated image)
           * If self.array() is loaded, then export the contents of self._array to the video file
           * If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video
           * If outfile==None or outfile==self.filename(), then overwrite the current filename 
           * If ignoreErrors=True, then exit gracefully.  Useful for chaining download().saveas() on parallel dataset downloads
           * Returns a new video object with this video filename, and a clean video filter chain
           * if flush=True, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel
           * framerate:  input framerate of the frames in the buffer, or the output framerate of the transcoded video.  If not provided, use framerate of source video
           * pause:  an integer in seconds to pause between loops of animated images
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
            
            elif self.isdownloaded() and self.isdirty():
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
            elif self.hasfilename() and not self.isdirty():
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
        return self.saveas(outfile=tempMP4())
    def savetemp(self):
        return self.savetmp()

    def ffplay(self):
        """Play the video file using ffplay.  If the video is loaded in memory, this will dump it to a temporary file first"""
        assert has_ffplay, '"ffplay" executable not found on path - Install from http://ffmpeg.org/download.html'
        if self.isloaded() or self.isdirty():
            f = tempMP4()
            warnings.warn('%s - Saving video to temporary file "%s" for ffplay ... ' % ('Video loaded into memory' if self.isloaded() else 'Dirty FFMPEG filter chain', f))
            v = self.saveas(f)
            cmd = 'ffplay "%s"' % v.filename()
            print('[vipy.video.play]: Executing "%s"' % cmd)
            os.system(cmd)
            os.remove(v.filename())  # cleanup
        elif self.hasfilename() or (self.hasurl() and self.download().hasfilename()):  # triggers download
            cmd = 'ffplay "%s"' % self.filename()
            print('[vipy.video.play]: Executing "%s"' % cmd)
            os.system(cmd)
        else:
            raise ValueError('Invalid video file "%s" - ffplay requires a video filename' % self.filename())
        return self
        
    def play(self, verbose=True, notebook=False, fps=30):
        """Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it"""

        if not self.isdownloaded() and self.hasurl():
            self.download()
        
        if notebook:
            # save to temporary video, this video is not cleaned up and may accumulate            
            try_import("IPython.display", "ipython"); import IPython.display
            if not self.hasfilename() or self.isloaded() or self.isdirty():
                v = self.saveas(tempMP4())                 
                warnings.warn('Saving video to temporary file "%s" for notebook viewer ... ' % v.filename())
                return IPython.display.Video(v.filename(), embed=True)
            return IPython.display.Video(self.filename(), embed=True)
        elif has_ffplay:
            return self.ffplay()            
        else:
            """Fallback player.  This can visualize videos without ffplay, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates."""
            fps = min(fps, self.framerate()) if fps is not None else self.framerate()
            assert fps > 0, "Invalid display framerate"
            with Stopwatch() as sw:            
                for (k,im) in enumerate(self.load() if self.isloaded() else self.stream()):
                    time.sleep(max(0, (1.0/self.framerate())*int(np.ceil((self.framerate()/fps))) - sw.since()))                                
                    im.show(figure=figure)
                    if vipy.globals.user_hit_escape():
                        break                    
            vipy.show.close(figure)
            return self


    def torch(self, startframe=0, endframe=None, length=None, stride=1, take=None, boundary='repeat', order='nchw', verbose=False, withslice=False, scale=1.0, withlabel=False, nonelabel=False):
        """Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().

           * Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.
           * Follows numpy slicing rules.  Optionally return the slice used if withslice=True
           * Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()           
           * order can be ['nchw', 'nhwc', 'cnhw'] for batchsize=n, channels=c, height=h, width=w
           * boundary can be ['repeat', 'strict', 'cyclic']
           * withlabel=True, returns tuple (t, labellist), where labellist is a list of tuples of activity labels occurring at the corresponding frame in the tensor
           * withslice=Trye, returnss tuple (t, (startframe, endframe, stride))
           * nonelabel=True, returns tuple (t, None) if withlabel=False 

        """
        try_import('torch'); import torch
        frames = self.load().numpy() if self.load().numpy().ndim == 4 else np.expand_dims(self.load().numpy(), 3)  # NxHxWx(C=1, C=3)
        assert boundary in ['repeat', 'strict', 'cyclic'], "Invalid boundary mode - must be in ['repeat', 'strict', 'cyclic']"

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
        elif boundary == 'cyclic' and j > len(frames):
            for d in range(j-len(frames)):
                frames = np.concatenate( (frames, np.expand_dims(frames[j % len(frames)], 0) ))            
        assert j <= len(frames), "invalid slice=%s for frame shape=%s - try setting boundary='repeat'" % (str((i,j,k)), str(frames.shape))
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
            labels = [sorted(tuple(self.activitylabels( (k%len(frames)) if boundary == 'cyclic' else min(k, len(frames)-1) ))) for f in range(i,j,k)]
            return (t, labels)
        elif nonelabel:
            return (t, None)
        else:
            return t

    def clone(self, flushforward=False, flushbackward=False, flush=False, flushfilter=False, rekey=False, flushfile=False, shallow=False, sharedarray=False):
        """Create deep copy of video object, flushing the original buffer if requested and returning the cloned object.
        Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned 
        object which can be used for encoding and will be garbage collected.
        
            * flushforward: copy the object, and set the cloned object array() to None.  This flushes the video buffer for the clone, not the object
            * flushbackward:  copy the object, and set the object array() to None.  This flushes the video buffer for the object, not the clone.
            * flush:  set the object array() to None and clone the object.  This flushes the video buffer for both the clone and the object.
            * flushfilter:  Set the ffmpeg filter chain to the default in the new object, useful for saving new videos
            * rekey: Generate new unique track ID and activity ID keys for this scene
            * shallow:  shallow copy everything (copy by reference), except for ffmpeg object
            * sharedarray:  deep copy of everything, except for pixel buffer which is shared

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
        if not self.isloaded() and self._startsec is not None:
            strlist.append('cliptime=(%1.2f,%1.2f)' % (self._startsec, self._endsec))
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
        if not self.isloaded() and self._startsec is not None:
            strlist.append('cliptime=(%1.2f,%1.2f)' % (self._startsec, self._endsec))            
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.hastracks():
            strlist.append('tracks=%d' % len(self._tracks))
        if self.hasactivities():
            strlist.append('activities=%d' % len(self._activities))
        return str('<vipy.video.scene: %s>' % (', '.join(strlist)))

    def frame(self, k, img=None):
        """Return vipy.image.Scene object at frame k"""
        assert isinstance(k, int) and k>=0, "Frame index must be non-negative integer"
        assert img is not None or (self.isloaded() and k<len(self)) or not self.isloaded(), "Invalid frame index %d - Indexing video by frame must be integer within (0, %d)" % (k, len(self)-1)

        img = img if img is not None else (self._array[k] if self.isloaded() else self.preview(k).array())
        dets = [t[k].clone().setattribute('trackindex', j) for (j, t) in enumerate(self.tracks().values()) if len(t)>0 and (t.during(k) or t.boundary()=='extend')]  # track interpolation (cloned) with boundary handling
        for d in dets:
            shortlabel = [(d.shortlabel(),'')]  # [(Noun, Verbing1), (Noun, Verbing2), ...]
            activityconf = [None]   # for display 
            for (aid, a) in self.activities().items():  # insertion order:  First activity is primary, next is secondary (not in confidence order) 
                if a.hastrack(d.attributes['trackid']) and a.during(k):
                    # Shortlabel is always displayed as "Noun Verbing" during activity (e.g. Person Carrying, Vehicle Turning)
                    # If noun is associated with more than one activity, then this is shown as "Noun Verbing1\nNoun Verbing2", with a newline separator 
                    if not any([a.shortlabel() == v for (n,v) in shortlabel]):
                        shortlabel.append( (d.shortlabel(), a.shortlabel()) )  # only show each activity once (even if repeated)
                        activityconf.append(a.confidence())
                    if 'activityid' not in d.attributes:
                        d.attributes['activityid'] = []                            
                    d.attributes['activityid'].append(a.id())  # for activity correspondence (if desired)
            d.shortlabel( '\n'.join([('%s %s' % (n,v)).strip() for (n,v) in shortlabel[0 if len(shortlabel)==1 else 1:]]))
            d.attributes['noun verb'] = shortlabel[0 if len(shortlabel)==1 else 1:]
            d.attributes['activityconf'] = activityconf[0 if len(shortlabel)==1 else 1:]
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
        framelist = [min(int(np.round(f))+startframe, len(self)-1) for f in np.linspace(0, len(self)-1, n)]
        imframes = [self.frame(k).maxmatte()  # letterbox or pillarbox
                    if (self.frame(k).boundingbox() is None) or (context is True and (j == 0 or j == (n-1))) else
                    self.frame(k).padcrop(self.frame(k).boundingbox().dilate(dilate).imclipshape(self.width(), self.height()).maxsquare().int()).mindim(mindim, interp='nearest')
                    for (j,k) in enumerate(framelist)]
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
    
    def tracklist(self):
        return list(self.tracks().values())  # triggers copy

    def actorid(self):
        #assert len(self.tracks()) == 1, "Actor ID only valid for scenes with a single track"
        return next(iter(self.tracks().keys()))   # Python >=3.6
        
    def actor(self):
        #assert len(self.tracks()) == 1, "Actor only valid for scenes with a single track"
        return next(iter(self.tracks().values()))   # Python >=3.6
        
    def primary_activity(self):
        return next(iter(self.activities().values()))   # Python >=3.6        

    def activities(self, activities=None, id=None):
        """Return mutable dictionary of activities.  All temporal alignment is relative to the current clip()."""
        if isinstance(self._activities, tuple):
            self._activities = {a.id():a for a in [vipy.activity.Activity.from_json(json.loads(s)) for s in self._activities]}  # on-demand
        if activities is None and id is None:
            return self._activities  # mutable dict
        elif id is not None:
            return self._activities[id]
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
        
           Usage:  Filter out all activities longer than 128 frames 
             vid = vid.activityfilter(lambda a: len(a)<128)

           Usage:  Filter out activities with category in set
             vid = vid.activityfilter(lambda a: a.category() in set(['category1', 'category2']))
       
        """
        self._activities = {k:a for (k,a) in self.activities().items() if f(a) == True}
        return self
        
    def trackfilter(self, f):
        """Apply lambda function f to each object and keep if filter is True"""
        self._tracks = {k:t for (k,t) in self.tracks().items() if f(t) == True} 
        return self

    def trackmap(self, f, strict=True):
        """Apply lambda function f to each activity"""
        self._tracks = {k:f(t) for (k,t) in self.tracks().items()}
        assert all([isinstance(t, vipy.object.Track) and (strict is False or not t.isdegenerate()) for (tk,t) in self.tracks().items()]), "Lambda function must return non-degenerate vipy.object.Track()"
        return self
        
    def activitymap(self, f):
        """Apply lambda function f to each activity"""
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

    def activitylabel(self):
        """Return an iterator over activity labels in each frame"""        
        endframe = max([a.endframe() for a in self.activitylist()]) if len(self.activities())>0 else 0
        for k in range(0, endframe):
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
        return trackid in self.tracks()

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
            self.tracks()[t.id()] = t  # by-reference
            return t.id()
        elif isinstance(obj, vipy.object.Track):
            if rangecheck and not obj.boundingbox().isinside(vipy.geometry.imagebox(self.shape())):
                obj = obj.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('Clipping track "%s" to image rectangle' % (str(obj)))
            self.tracks()[obj.id()] = obj  # by-reference
            return obj.id()
        elif isinstance(obj, vipy.activity.Activity):
            if rangecheck and obj.startframe() >= obj.endframe():
                raise ValueError("Activity '%s' has invalid (startframe, endframe)=(%d, %d)" % (str(obj), obj.startframe(), obj.endframe()))
            self.activities()[obj.id()] = obj  # by-reference, activity may have no tracks
            return obj.id()
        elif (istuple(obj) or islist(obj)) and len(obj) == 4 and isnumber(obj[0]):
            assert self._currentframe is not None, "add() for obj=xywh must be added during frame iteration (e.g. for im in video: )"
            t = vipy.object.Track(category=category, keyframes=[self._currentframe], boxes=[vipy.geometry.BoundingBox(xywh=obj)], boundary='strict', attributes=attributes)
            if rangecheck and not t.boundingbox().isinside(vipy.geometry.imagebox(self.shape())):
                t = t.imclip(self.width(), self.height())  # try to clip it, will throw exception if all are bad 
                warnings.warn('Clipping track "%s" to image rectangle' % (str(t)))
            self.tracks()[t.id()] = t  # by-reference
            return t.id()
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
        """Return JSON encoded string of this object"""
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
                ';'.join([a.category() for a in self.activities()[d.attributes['activityid']]] if 'activityid' in d.attributes else ''), # semicolon separated activity ID assocated with track
                d.xmin(), d.ymin(), d.width(), d.height(),   # bounding box
                d.attributes['trackid'],  # globally unique track ID
                ';'.join([a.id() for a in self.activities()[d.attributes['activityid']]] if 'activityid' in d.attributes else '')) # semicolon separated activity ID assocated with track
               for (k,im) in enumerate(self) for d in im.objects()]
        csv = [('# video_filename', 'frame_number', 'object_category', 'object_shortlabel', 'activity categories(;)', 'xmin', 'ymin', 'width', 'height', 'track_id', 'activity_ids(;)')] + csv
        return writecsv(csv, outfile) if outfile is not None else csv


    def framerate(self, fps=None):
        """Change the input framerate for the video and update frame indexes for all annotations

           * NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()
        """
        
        if fps is None:
            return self._framerate
        else:
            assert not self.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"
            if self._startframe is not None and self._endframe is not None:
                (self._startframe, self._endframe) = (int(round(self._startframe * (fps/self._framerate))), int(round(self._endframe * (fps/self._framerate))))
            self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
            self._tracks = {k:t.framerate(fps) for (k,t) in self.tracks().items()}
            self._activities = {k:a.framerate(fps) for (k,a) in self.activities().items()}        
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
        return [vid.clone().setattribute('activityindex', k).activities(pa).tracks(t) for (k,(pa,t)) in enumerate(zip(activities, tracks))]

    def tracksplit(self):
        """Split the scene into k separate scenes, one for each track.  Each scene starts at frame 0 and is a shallow copy of self containing exactly one track.  Use clone() to create a deep copy if needed."""
        return [self.clone(shallow=True).tracks(t).activityfilter(lambda a: a.hastrack(tk)) for (tk,t) in self.tracks().items()] 

    def trackclip(self):
        """Split the scene into k separate scenes, one for each track.  Each scene starts and ends when the track starts and ends"""
        return [t.clip(t.track(t.actorid()).startframe(), t.track(t.actorid()).endframe()) for t in self.tracksplit()]
    
    def activityclip(self, padframes=0, multilabel=True):
        """Return a list of vipy.video.Scene() each clipped to be temporally centered on a single activity, with an optional padframes before and after.  
           
           * The Scene() category is updated to be the activity, and only the objects participating in the activity are included.
           * Activities are returned ordered in the temporal order they appear in the video.
           * The returned vipy.video.Scene() objects for each activityclip are clones of the video, with the video buffer flushed.
           * Each activityclip() is associated with each activity in the scene, and includes all other secondary activities that the objects in the primary activity also perform (if multilabel=True).  See activityclip().labels(). 
           * Calling activityclip() on activityclip(multilabel=True) can result in duplicate activities, due to the overlapping secondary activities being included in each clip.  Be careful. 
        """
        vid = self.clone(flushforward=True)
        if any([(a.endframe()-a.startframe()) <= 0 for a in vid.activities().values()]):
            warnings.warn('Filtering invalid activity clips with degenerate lengths: %s' % str([a for a in vid.activities().values() if (a.endframe()-a.startframe()) <= 0]))
        primary_activities = sorted([a.clone() for a in vid.activities().values() if (a.endframe()-a.startframe()) > 0], key=lambda a: a.startframe())   # only activities with at least one frame, sorted in temporal order
        tracks = [ [t.clone() for (tid, t) in vid.tracks().items() if a.hastrack(t)] for a in primary_activities]  # tracks associated with each primary activity (may be empty)
        secondary_activities = [[sa.clone() for sa in primary_activities if (sa.id() != pa.id() and pa.hasoverlap(sa) and (len(T)==0 or any([sa.hastrack(t) for t in T])))] for (pa, T) in zip(primary_activities, tracks)]  # overlapping secondary activities that includes any track in the primary activity
        secondary_activities = [sa if multilabel else [] for sa in secondary_activities]  
        vid._activities = {}  # for faster clone
        vid._tracks = {}      # for faster clone
        padframes = padframes if istuple(padframes) else (padframes,padframes)
        return [vid.clone()
                .activities([pa]+sa)
                .tracks(t)
                .clip(startframe=max(pa.startframe()-padframes[0], 0), endframe=(pa.endframe()+padframes[1]))
                .category(pa.category())
                .setattribute('activityindex',k)
                for (k,(pa,sa,t)) in enumerate(zip(primary_activities, secondary_activities, tracks))]

    def noactivityclip(self, label=None, strict=True, padframes=0):
        """Return a list of vipy.video.Scene() each clipped on a track segment that has no associated activities.  
        
           * Each clip will contain exactly one activity "Background" which is the interval for this track where no activities are occurring
           * Each clip will be at least one frame long
           * strict=True means that background can only occur in frames where no tracks are performing any activities.  This is useful so that background is not constructed from secondary objects.
           * struct=False means that background can only occur in frames where a given track is not performing any activities. 
           * label=str: The activity label to give the background activities.  Defaults to the track category (lowercase)
           * padframes=0:  The amount of temporal padding to apply to the clips before and after in frames

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
        """The trackbox is the union of all track bounding boxes in the video, or None if there are no tracks"""
        boxes = [t.clone().boundingbox() for t in self.tracklist()]
        boxes = [bb.dilate(dilate) for bb in boxes if bb is not None]
        return boxes[0].union(boxes[1:]) if len(boxes) > 0 else None

    def framebox(self):
        """Return the bounding box for the image rectangle, requires preview() to get frame shape"""
        return vipy.geometry.BoundingBox(xmin=0, ymin=0, width=self.width(), height=self.height())

    def trackcrop(self, dilate=1.0, maxsquare=False, zeropad=True):
        """Return the trackcrop() of the scene which is the crop of the video using the trackbox().  
         
           * If there are no tracks, return None.  
           * if zeropad=True, the zero pad the crop if it is outside the image rectangle, otherwise return only valid pixels

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
        """Clip the video to between (startframe, endframe).  This clip is relative to clip() shown by __repr__().  Return a clone of the video for idempotence"""
        assert (endframe is None or startframe <= endframe) and startframe >= 0, "Invalid start and end frames (%s, %s)" % (str(startframe), str(endframe))

        v = self.clone()
        if not v.isloaded():
            # -- Copied from super().clip() to allow for clip on clone (for indempotence)
            # -- This code copy is used to avoid super(Scene, self.clone()) which screws up class inheritance for iPython reload
            assert not v.isloaded(), "Filters can only be applied prior to load() - Try calling flush() first"            
            timestamp_in_seconds = ((v._startframe if v._startframe is not None else 0)+startframe)/float(v.framerate())
            v._update_ffmpeg_seek(timestamp_in_seconds)
            if endframe is not None:
                v._ffmpeg = v._ffmpeg.trim(start_frame=0, end_frame=(endframe-startframe))
            v._ffmpeg = v._ffmpeg.setpts('PTS-STARTPTS')  # reset timestamp to 0 after trim filter            
            v._startframe = startframe if v._startframe is None else v._startframe + startframe  # for __repr__ only
            v._endframe = (v._startframe + (endframe-startframe)) if endframe is not None else v._endframe  # for __repr__ only
            # -- end copy
        else:
            endframe = endframe if endframe is not None else len(self._array)
            v._array = self._array[startframe:endframe]
            (v._startframe, v._endframe) = (0, endframe-startframe)
        v._tracks = {k:t.offset(dt=-startframe).truncate(startframe=0, endframe=endframe-startframe) for (k,t) in v.tracks().items()}   # may be degenerate
        v._activities = {k:a.offset(dt=-startframe).truncate(startframe=0, endframe=endframe-startframe) for (k,a) in v.activities().items()}  # may be degenerate
        return v.trackfilter(lambda t: len(t)>0).activityfilter(lambda a: len(a)>0)  # remove degenerate tracks and activities

    def cliptime(self, startsec, endsec):
        raise NotImplementedError('FIXME: use clip() instead for now')
            
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
        (H,W) = self.shape()  # yuck, need to get image dimensions before filter
        sy = rows / float(H) if rows is not None else cols / float(W)
        sx = cols / float(W) if cols is not None else rows / float(H)
        self._tracks = {k:t.scalex(sx) for (k,t) in self.tracks().items()}
        self._tracks = {k:t.scaley(sy) for (k,t) in self.tracks().items()}
        super().resize(rows, cols)
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
        """Spatially rescale the scene by a constant scale factor"""
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
            oc = oc.trackfilter(lambda t: ((not t.isdegenerate()) and len(t)>0))

            # Merge other tracks into selfclone: one-to-many mapping from self to other
            merged = {}  # dictionary mapping trackid in other to the trackid in self, each track in other can be merged at most once
            for ti in sorted(sc.tracklist(), key=lambda t: len(t), reverse=True):  # longest to shortest
                for tj in sorted(oc.tracklist(), key=lambda t: len(t), reverse=True):  
                    if ti.category() == tj.category() and (tj.id() not in merged) and tj.segment_percentilecover(sc.track(ti.id()), percentile=percentilecover, samples=percentilesamples) > spatial_iou_threshold:  # mean framewise overlap during overlapping segment of two tracks
                        sc.tracks()[ti.id()] = sc.track(ti.id()).union(tj, overlap=overlap)  # merge duplicate/fragmented tracks from other into self, union() returns clone
                        merged[tj.id()] = ti.id()  
                        print('[vipy.video.union]: merging track "%s"(id=%s) + "%s"(id=%s) for scene "%s"' % (str(ti), str(ti.id()), str(tj), str(tj.id()), str(sc)))                        
            oc.trackfilter(lambda t: t.id() not in merged)  # remove duplicate other track for final union

            # Merge other activities into selfclone: one-to-one mapping
            for (i,j) in merged.items():  # i=id of other, j=id of self
                oc.activitymap(lambda a: a.replaceid(i, j) if a.hastrack(i) else a)  # update track IDs referenced in activities for merged tracks
            for (i,ai) in sc.activities().items():
                for (j,aj) in oc.activities().items():
                    if ai.category() == aj.category() and set(ai.trackids()) == set(aj.trackids()) and ai.temporal_iou(aj) > temporal_iou_threshold:
                        oc.activityfilter(lambda a: a.id() != j)  # remove duplicate activity from final union
                        
            # Union
            sc.tracks().update(oc.tracks())
            sc.activities().update(oc.activities())

        # Final union of unique tracks/activities
        if track:
            self.tracks(sc.tracklist())  # union of tracks only
        if activity:
            self.activities(sc.activitylist())  # union of activities only: may reference tracks not in self of track=False
        return self        

    def annotate(self, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], mutator=None, timestamp=None, timestampcolor='black', timestampfacecolor='white', outfile=None):
        """Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, pixels in this video will now contain the overlay
        This function does not play the video, it only generates an annotation video frames.  Use show() which is equivalent to annotate().saveas().play()
        
            * In general, this function should not be run on very long videos, as it requires loading the video framewise into memory, try running on clips instead.
            * For long videos, a btter strategy given a video object vo with an output filename which will use a video stream for annotation

        """
        if verbose:
            print('[vipy.video.annotate]: Annotating video ...')  
    
        f_mutator = mutator if mutator is not None else lambda im,k: im
        f_timestamp = (lambda k: '%s %d' % (vipy.util.clockstamp(), k)) if timestamp is True else timestamp

        if outfile is None:        
            assert self.load().isloaded(), "Load() failed"
            imgs = [f_mutator(self[k], k).savefig(fontsize=fontsize,
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
            n = self.duration_in_frames_of_videofile()
            vo = vipy.video.Video(filename=outfile, framerate=self.framerate())
            with vo.stream(overwrite=True) as so:
                for (k,im) in enumerate(self.stream()):
                    so.write(f_mutator(im, k).savefig(fontsize=fontsize,
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


    def show(self, outfile=None, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], notebook=False, timestamp=None, timestampcolor='black', timestampfacecolor='white'):
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
    

    def fastshow(self, outfile=None, verbose=True, fontsize=10, captionoffset=(0,0), textfacecolor='white', textfacealpha=1.0, shortlabel=True, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, categories=None, nocaption=False, nocaption_withstring=[], figure=1, fps=None, timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None):
        """Faster show using interative image show for annotated videos.  This can visualize videos before video rendering is complete, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates."""
        fps = min(fps, self.framerate()) if fps is not None else self.framerate()
        assert fps > 0, "Invalid display framerate"
        f_timestamp = (lambda k: '%s %d' % (vipy.util.clockstamp(), k)) if timestamp is True else timestamp
        f_mutator = mutator if mutator is not None else lambda im,k: im        
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
                
                if vipy.globals.user_hit_escape():
                    break
        vipy.show.close(figure)
        return self

    def thumbnail(self, outfile=None, frame=0, fontsize=10, nocaption=False, boxalpha=0.25, dpi=200, textfacecolor='white', textfacealpha=1.0):
        """Return annotated frame=k of video, save annotation visualization to provided outfile if provided, otherwise return vipy.image.Scene"""
        im = self.frame(frame, img=self.preview(framenum=frame).array())
        return im.savefig(outfile=outfile, fontsize=fontsize, nocaption=nocaption, boxalpha=boxalpha, dpi=dpi, textfacecolor=textfacecolor, textfacealpha=textfacealpha) if outfile is not None else im
    
    def stabilize(self, show=False):
        """Background stablization using flow based stabilization masking foreground region.  This will output a video with all frames aligned to the first frame, such that the background is static."""
        from vipy.flow import Flow  # requires opencv
        return Flow().stabilize(self.clone(), residual=True, show=show)
    
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
        """Cast the object to a vipy.video.Video class"""
        self.__class__ = vipy.video.Video
        return self

    def merge_tracks(self, dilate_height=2.0, dilate_width=2.0, framedist=5):
        """Merge tracks if a track endpoint dilated by a fraction overlaps exactly one track startpoint, and the endpoint and startpoint are close enough together temporally.
        
           * This is useful for continuing tracking when the detection framerate was too low and the assignment falls outside the measurement gate.
           * This will not work for complex scenes, as it assumes that there is exactly one possible continuation for a track.  
        
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

    def assign(self, frame, dets, minconf=0.2, maxhistory=5, activityiou=0.5, trackcover=0.2, trackconfsamples=4, gate=0, activitymerge=True):
        """Assign a list of vipy.object.Detections at frame k to scene by greedy track association. In-place update.
        
           * miniou [float]: the minimum temporal IOU for activity assignment
           * minconf [float]: the minimum confidence for a detection to be considered as a new track
           * maxhistory [int]:  the maximum propagation length of a track with no measurements, the frame history ised for velocity estimates  
           * trackconfsamples [int]:  the number of uniformly spaced samples along a track to compute a track confidence
           * gate [int]: the gating distance in pixels used for assignment of fast moving detections.  Useful for low detection framerates if a detection does not overlap with the track.
           * trackcover [float]: the minimum cover necessary for assignment of a detection to a track
           * activitymerge [bool]: if true, then merge overlapping activity detections of the same track and category, otherwise each activity detection is added as a new detection

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
                        self.track(t.id()).add(frame, d.clone())  # track assignment! (clone required)
                        assigned.add(t.id())  # cannot assign again to this track
                        assigned.add(d.id())  # mark detection as assigned
                
            # Track spawn from unassigned and unexplained detections 
            for (j,d) in enumerate(objdets):                
                if (d.id() not in assigned):
                    if (d.confidence() >= minconf and not any([t.linear_extrapolation(frame, dt=maxhistory, shape=False).maxcover(d, otherarea=detarea[j]) >= 0.7 for (i,(t,ti)) in enumerate(t_ref) if t.category() == d.category()])):
                        gated = [(t, t.linear_extrapolation(frame, dt=maxhistory, shape=False)) for (t,ti) in t_ref if (t.id() not in assigned and t.category() == d.category())] if gate>0 else []
                        gated = sorted([(t, ti) for (t, ti) in gated if ti.hasintersection(d, gate=gate)], key=lambda x: d.sqdist(x[1]))
                        if len(gated) > 0:
                            self.track(gated[0][0].id()).add(frame, d.clone())  # track assignment! (clone required)
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
                activitydets.sort(key=lambda a: a.startframe())  # in-place
                for a in self.activities().values():  # assumed sorted
                    for d in activitydets: 
                        if (d.id() not in assigned) and (a.category() == d.category()) and (a.actorid() == d.actorid()) and a.hasoverlap(d, activityiou): 
                            a.union(d)  # activity assignment with maximum confidence
                            assigned.add(d.id())
                        
            # Activity construction from unassigned detections
            for d in activitydets:
                if d.id() not in assigned:
                    self.add(d.clone())  

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
                                keyframes=[0, np.random.randint(50,100), 150],
                                boxes=[vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,32)]

    activities = [vipy.activity.Activity(label='activity%d' % k, shortlabel='a%d' % k, tracks=[tracks[j].id() for j in [np.random.randint(32)]], startframe=np.random.randint(50,99), endframe=np.random.randint(100,150)) for k in range(0,32)]   
    return Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)


def RandomSceneActivity(rows=None, cols=None, frames=256):
    """Return a random loaded vipy.video.Scene, useful for unit testing"""    
    v = RandomVideo(rows, cols, frames)
    (rows, cols) = v.shape()
    tracks = [vipy.object.Track(label=['Person','Vehicle','Object'][k], shortlabel='track%d' % k, boundary='strict', 
                                keyframes=[0, np.random.randint(50,100), np.random.randint(50,150)],
                                boxes=[vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2)),
                                       vipy.object.Detection(xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                 width=np.random.randint(16,cols//2), height=np.random.randint(16,rows//2))]) for k in range(0,3)]

    activities = [vipy.activity.Activity(label='Person Carrying', shortlabel='Carry', tracks=[tracks[0].id(), tracks[1].id()], startframe=np.random.randint(20,50), endframe=np.random.randint(70,100))]   
    ims = Scene(array=v.array(), colorspace='rgb', category='scene', tracks=tracks, activities=activities)

    return ims
    
def EmptyScene():
    """Return an empty scene""" 
    return vipy.video.Scene(array=np.zeros((1,1,1,3), dtype=np.uint8))
