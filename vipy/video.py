import os
from vipy.util import isnumpy, quietprint, isstring, isvideo, tempcsv, imlist, remkdir, filepath, filebase, tempMP4, isurl, isvideourl, templike, tempjpg
from vipy.image import Image, ImageCategory, ImageDetection
from vipy.show import savefig, figure
import vipy.downloader
import copy
import numpy as np
import ffmpeg
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib
import io


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

    def clip(self):
        pass

    def show(self):                
        for im in imframes:
            buf = io.BytesIO()
            im.show(figure=1)
            savefig(buf, format='png')


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
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"               
        self._ffmpeg = self._ffmpeg.trim(start_frame=startframe, end_frame=endframe) \
                                   .setpts ('PTS-STARTPTS')
        return self
    
    def trim(self, startframe, endframe):
        """Alias for clip"""
        return self.clip(startframe, endframe)

    def rot90cw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"
        self._ffmpeg = self._ffmpeg.filter('transpose', 1)
        return self

    def rot90ccw(self):
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"
        self._ffmpeg = self._ffmpeg.filter('transpose', 2)        
        return self
    
    def rescale(self, s):
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"        
        self._ffmpeg = self._ffmpeg.filter('scale', 'iw*%1.2f' % s, 'ih*%1.2f' % s)
        return self
        
    def resize(self, rows=None, cols=None):
        if rows is None and cols is None:
            return self 
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"               
        self._ffmpeg = self._ffmpeg.filter('scale', cols if cols is not None else -1, rows if rows is not None else -1)
        return self

    def framerate(self, fps):
        assert not self.isloaded(), "Filters can only be applied prior to loading the video"
        self._ffmpeg = self._ffmpeg.filter('fps', fps=fps, round='up')
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



class Track(Video):
    def __init__(self, url=None, filename=None, framerate=30, rot90cw=False, rot90ccw=False, attributes=None, track=None):
        super(Track, self).__init__(url=url, filename=filename, framerate=framerate, rot90cw=rot90cw, rot90ccw=rot90ccw, attributes=attributes)
        self._track = track

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, frames=%d" % (self._array[0].shape[0], self._array[0].shape[1], len(self._array)))        
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        if self._track is not None:
            strlist.append('keyframes="%s"' % str(self._track.keyframes()))
        return str('<vipy.video.Track: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        self.load()
        if k >= 0 and k < len(self):
            t = self._track[k]
            return ImageDetection(array=self._array[k], colorspace='rgb', category=t.category(), bbox=t)            
        else:
            raise ValueError('Invalid frame index %d ' % k)
    
    def __iter__(self):
        self.load()
        for k in range(0, len(self)):
            yield self.__getitem__(k)
        
    def show(self):
        assert self.isloaded(), "Show requires that the video is loaded"
        vid = self.load().clone()
        import matplotlib.pyplot as plt
        import PIL.Image
        (W, H) = (None, None)
        for (k,im) in enumerate(vid):
            print(im)
            im.show(figure=1)
            if W is None or H is None:
                (W,H) = plt.figure(1).canvas.get_width_height()                        
            
            buf = io.BytesIO()
            plt.figure(1).canvas.print_raw(buf)
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape( (H, W, 4))
            vid._array[k] = np.array(PIL.Image.fromarray(img).convert('RGB')).copy()

        vid.saveas('/Users/jba3139/Desktop/vipy.mp4')
        return self

    
    
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

    
