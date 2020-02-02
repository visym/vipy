import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase, tempMP4, isurl, isvideourl, templike
from vipy.image import Image, ImageCategory, ImageDetection
import vipy.downloader
import copy
import numpy as np
import ffmpeg
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib


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

    def clip(self):
        pass
        
    
class Video(object):
    def __init__(self, url=None, filename=None, startframe=0, framerate=30, rot90cw=False, rot90ccw=False, attributes=None):
        self._ignoreErrors = False
        self._url = url
        self._filename = filename
        self._framerate = framerate
        self._array = None
        self.attributes = attributes if attributes is not None else {}
        
        if url is not None:
            assert isurl(url), 'Invalid URL "%s" ' % url
        assert filename is not None or url is not None, 'Invalid constructor'
        
        
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

    def tonumpy(self):
        return self.array()

    def numpy(self):
        return self.tonumpy()

    def flush(self):
        self._array = None
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
        if self._filename is None:
            if 'VIPY_CACHE' in os.environ:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._url))
            elif isvideourl(self._url):
                self._filename = templike(self._url)
            else:
                self._filename = tempMP4()  # guess MP4 for URLs with no file extension

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
        probe = ffmpeg.probe(self.filename())
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return (width, height)
    
    def load(self):
        (width, height) = self.shape()
        (out, err) = ffmpeg.input(self.filename()) \
                           .output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                           .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return self

    def clip(self, startframe, endframe):
        """Load a video clip betweeen start and end frames"""
        assert(startframe < endframe)
        (width, height) = self.shape()        
        (out, err) = ffmpeg.input(self.filename()) \
                           .trim(start_frame=startframe, end_frame=endframe) \
                           .setpts ('PTS-STARTPTS') \
                           .output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                           .run(capture_stdout=True)
        self._array = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return self
    
    def pptx(self, outfile):
        pass

    def saveas(self, outfile, framerate=30, vcodec='libx264'):
        (n, height, width, channels) = self._array.shape
        process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)) \
                        .output(outfile, pix_fmt='yuv420p', vcodec=vcodec, r=framerate) \
                        .overwrite_output() \
                        .global_args('-loglevel', 'error') \
                        .run_async(pipe_stdin=True)
        
        for frame in self._array:
            process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()
        return outfile


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

    
