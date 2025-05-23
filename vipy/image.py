import os
import PIL
import PIL.Image
import PIL.ImageFilter
import platform
import dill
import vipy.show
import vipy.globals
from vipy.globals import log, cache
from vipy.util import isnumpy, isurl, isimageurl, \
    fileext, tempimage, mat2gray, imwrite, imwritegray, \
    tempjpg, filetail, isimagefile, remkdir, hasextension, \
    try_import, tolist, islistoflists, istupleoftuples, isstring, \
    islist, isnumber, isnumpyarray, string_to_pil_interpolation, toextension, iswebp
from vipy.geometry import BoundingBox, imagebox
import vipy.object
from vipy.object import greedy_assignment
import vipy.downloader
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib
import copy
from copy import deepcopy
import numpy as np
import shutil
import io
import matplotlib.pyplot as plt
import base64
import types
import hashlib
import time
import math


try:
    import torch  # pre-import
except:
    pass  # will throw exception on vipy.image.Image.torch

try:
    import ujson as json  # faster
except ImportError:        
    import json  # fastish
    

class Image(object):
    """vipy.image.Image class
    
    The vipy image class provides a fluent, lazy interface for representing, transforming and visualizing images.
    The following constructors are supported:

    ```python
    im = vipy.image.Image(filename="/path/to/image.ext")
    ```
    
    All image file formats that are readable by PIL are supported here.

    ```python
    im = vipy.image.Image(url="http://domain.com/path/to/image.ext")
    ```
    
    The image will be downloaded from the provided url and saved to a temporary filename.
    The environment variable VIPY_CACHE controls the location of the directory used for saving images, otherwise this will be saved to the system temp directory.

    ```python
    im = vipy.image.Image(url="http://domain.com/path/to/image.ext", filename="/path/to/new/image.ext")
    ```

    The image will be downloaded from the provided url and saved to the provided filename.
    The url() method provides optional basic authentication set for username and password

    ```python
    im = vipy.image.Image(array=img, colorspace='rgb')
    ```

    The image will be constructed from a provided numpy array 'img', with an associated colorspace.  The numpy array and colorspace can be one of the following combinations:

    - 'rgb': uint8, three channel (red, green, blue)
    - 'rgba':  uint8, four channel (rgb + alpha)
    - 'bgr': uint8, three channel (blue, green, red), such as is returned from cv2.imread()
    - 'bgra':  uint8, four channel
    - 'hsv':  uint8, three channel (hue, saturation, value)
    - 'lum;:  uint8, one channel, luminance (8 bit grey level)
    - 'grey':  float32, one channel in range [0,1] (32 bit intensity)
    - 'float':  float32, any channel in range [-inf, +inf]
    
    The most general colorspace is 'float' which is used to manipulate images prior to network encoding, such as applying bias. 
    
    Args:
        filename: a path to an image file that is readable by PIL
        url:  a url string to an image file that is readable by PIL
        array: a numpy array of type uint8 or float32 of shape HxWxC=height x width x channels
        colorspace:  a string in ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']
        attributes:  a python dictionary that is passed by reference to the image.  This is useful for encoding metadata about the image.  Accessible as im.attributes

    Returns:
        A `vipy.image.Image` object

    """

    def __init__(self, filename=None, url=None, array=None, colorspace=None, attributes=None):
        # Private attributes
        self._filename = None   # Local filename
        self._url = None        # URL to download
        self._loader = None     # lambda function to load an image, set with loader() method
        self._array = None
        self._colorspace = None
        
        # Initialization
        self._filename = filename
        if url is not None:
            assert isurl(url), 'Invalid URL'
        self._url = url
        if array is not None:
            assert isnumpy(array), 'Invalid Array - Type "%s" must be np.array()' % (str(type(array)))
        self.array(array)  # shallow copy

        # Colorspace guesses:
        if not colorspace:
            # Guess RGB colorspace if three channel uint8 if colorspace is not provided
            colorspace = 'rgb' if (self.isloaded() and self.channels() == 3 and self._array.dtype == np.uint8) else colorspace

            # Guess LUM colorspace if three channel uint8 if colorspace is not provided
            colorspace = 'lum' if (self.isloaded() and self.channels() == 1 and self._array.dtype == np.uint8) else colorspace
            
            # Guess float colorspace if array is float32 and colorspace is not provided        
            colorspace = 'float' if (self.isloaded() and self._array.dtype == np.float32) else colorspace
            
        self.colorspace(colorspace)
        
        # Public attributes: passed in as a dictionary
        self.attributes = {} 
        if attributes is not None:
            assert isinstance(attributes, dict), "Attributes must be dictionary"
            self.attributes = attributes

    @classmethod
    def cast(cls, im):
        """Typecast the conformal vipy.image object im as `vipy.image.Image`.
        
        This is useful for downcasting `vipy.image.Scene` or `vipy.image.ImageDetection` down to an image.

        ```python
        ims = vipy.image.RandomScene()
        im = vipy.image.Image.cast(im)
        ```

        """
        assert isinstance(im, vipy.image.Image), "Invalid input - must derive from vipy.image.Image"
        im.__class__ = vipy.image.Image
        return im

    @classmethod
    def from_dict(cls, d):
        d = {k.lstrip('_'):v for (k,v) in d.items()}  # prettyjson (remove "_" prefix to attributes)                
        return cls(filename=d['filename'],
                   url=d['url'],
                   array=np.array(d['array'], dtype=np.uint8) if d['array'] is not None else None,
                   colorspace=d['colorspace'],
                   attributes=d['attributes'])
        
    
    @classmethod
    def from_json(cls, s):
        """Import the JSON string s as an `vipy.image.Image` object.

        Args:
            s: json encoded string
        
        This will perform a round trip such that im1 == im2

        ```python
        im1 = vupy.image.RandomImage()
        im2 = vipy.image.Image.from_json(im1.json())
        assert im1 == im2
        ```

        Note: to construct from non-encoded json (e.g. a dict prior to dumps), use from_dict
        
        """
        return cls.from_dict(json.loads(s) if not isinstance(s, dict) else s)
    
    def __eq__(self, other):
        """Images are equivalent if they have the same filename, url and array"""
        return isinstance(other, Image) and other.filename()==self.filename() and other.url()==self.url() and np.all(other.array() == self.array())

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Yield single image for consistency with videos"""
        yield self

    def __len__(self):
        """Images have length 1 always"""
        return 1
    
    def __array__(self):
        """Called on np.array(self) for custom array container, (requires numpy >=1.16)"""
        return self.numpy()
    
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self._array.shape[0], self._array.shape[1], self.colorspace()))
        if self.colorspace() == 'float':
            strlist.append('channels=%d' % self.channels())
        if self.filename() is not None:
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        return str('<vipy.image: %s>' % (', '.join(strlist)))

    def sanitize(self):
        """Remove all private keys from the attributes dictionary.
        
        The attributes dictionary is useful storage for arbitrary (key,value) pairs.  However, this storage may contain sensitive information that should be scrubbed from the media before serialization.  As a general rule, any key that is of the form '__keyname' prepended by two underscores is a private key.  This is analogous to private or reserved attributes in the python lanugage.  Users should reserve these keynames for those keys that should be sanitized and removed before any serialization of this object.
        
        ```python
        assert self.setattribute('__mykey', 1).sanitize().hasattribute('__mykey') == False
        ```

        """
        self.attributes = {k:v for (k,v) in self.attributes.items() if not k.startswith('__')} if isinstance(self.attributes, dict) else self.attributes
        return self
    
    def print(self, prefix='', sleep=None):
        """Print the representation of the image and return self with an optional sleep=n seconds
        
        Useful for debugging in long fluent chains.
        """
        print(prefix+self.__repr__())
        if sleep is not None:
            assert isinstance(sleep, int) and sleep > 0, "Sleep must be a non-negative integer number of seconds"
            time.sleep(sleep)
        return self

    def exif(self, extended=False):
        """Return the EXIF meta-data in filename as a dictionary.  Included non-base EXIF data if extended=True.  Returns empty dictionary if no EXIF exists.  Triggers download."""

        d = {}
        if self.download().hasfilename():
            exif = PIL.Image.open(self.filename()).getexif()
            if exif is not None:
                d = {PIL.ExifTags.TAGS[k]:v for (k,v) in exif.items() if k in PIL.ExifTags.TAGS}

            if extended:
                for ifd_id in PIL.ExifTags.IFD:
                    try:
                        ifd = exif.get_ifd(ifd_id)                    
                        if ifd_id == PIL.ExifTags.IFD.GPSInfo:
                            resolve = PIL.ExifTags.GPSTAGS
                        else:
                            resolve = PIL.ExifTags.TAGS
                            
                            for k, v in ifd.items():
                                tag = resolve.get(k, k)
                                d[tag] = v
                    except KeyError:
                        pass
        return d
    
    def tile(self, tilewidth, tileheight, overlaprows=0, overlapcols=0):
        """Generate an image tiling.
        
        A tiling is a decomposition of an image into overlapping or non-overlapping rectangular regions.  

        Args:
            tilewidth: [int] the image width of each tile
            tileheight: [int] the image height of each tile
            overlaprows: [int] the number of overlapping rows (height) for each tile
            overlapcols: [int] the number of overlapping width (width) for each tile
    
        Returns:
            A list of `vipy.image.Image` objects such that each image is a single tile and the set of these tiles forms the original image
            Each image in the returned list contains the 'tile' attribute which encodes the crop used to create the tile.

        .. note:: 
            - `vipy.image.Image.tile` can be undone using `vipy.image.Image.untile`
            - The identity tiling is im.tile(im.widht(), im.height(), overlaprows=0, overlapcols=0)
            - Ragged tiles outside the image boundary are zero padded
            - All annotations are updated properly for each tile, when the source image is `vipy.image.Scene`
        """
        assert tilewidth > 0 and tileheight > 0 and overlaprows >= 0 and overlapcols >= 0, "Invalid input"
        assert self.width() >= tilewidth-overlapcols and self.height() >= tileheight-overlaprows, "Invalid input" 
        bboxes = [BoundingBox(xmin=i, ymin=j, width=min(tilewidth, self.width()-i), height=min(tileheight, self.height()-j)) for i in range(0, self.width()-overlapcols, tilewidth-overlapcols) for j in range(0, self.height()-overlaprows, tileheight-overlaprows)]
        return [self.clone(shallow=True, attributes=True).setattribute('tile', {'crop':bb, 'shape':self.shape()}).crop(bb) for bb in bboxes]

    def union(self, other):
        """No-op for `vipy.image.Image`"""
        return self
    
    @classmethod
    def untile(cls, imlist):
        """Undo an image tiling and recreate the original image.

        ```python
        tiles = im.tile(im.width()/2, im.height()/2, 0, 0)
        imdst = vipy.image.Image.untile(tiles)
        assert imdst == im
        ```

        Args:
            imlist: this must be the output of `vipy.image.Image.tile`
        
        Returns:
            A  new `vipy.image.Image` object reconstructed from the tiling, such that this is equivalent to the input to vipy.image.Image.tile` 
        
        .. note:: All annotations are updated properly for each tile, when the source image is `vipy.image.Scene`
        """
        assert all([isinstance(im, vipy.image.Image) and im.hasattribute('tile') for im in imlist]), "invalid image tile list"        
        imc = None
        for im in imlist:
            if imc is None:
                imc = im.clone(shallow=True).array(np.zeros( (im.attributes['tile']['shape'][0], im.attributes['tile']['shape'][1], im.channels()), dtype=np.uint8))                
            imc = imc.splat(im.array(im.attributes['tile']['crop'].clone().to_origin().int().crop(im.array())), im.attributes['tile']['crop'])
            if hasattr(im, 'objectmap'):
                im.objectmap(lambda o: o.set_origin(im.attributes['tile']['crop']))  # FIXME: only for Scene()
            imc = imc.union(im)
        return imc
    
    def uncrop(self, bb, shape):
        """Uncrop using provided bounding box and zeropad to shape=(Height, Width).

        An uncrop is the inverse operation for a crop, which preserves the cropped portion of the image in the correct location and replaces the rest with zeros out to shape.
    
        ```python
        im = vipy.image.RandomImage(128, 128)
        bb = vipy.geometry.BoundingBox(xmin=0, ymin=0, width=64, height=64)
        uncrop = im.crop(bb).uncrop(bb, shape=(128,128))
        ```

        Args:
            bb: [`vipy.geometry.BoundingBox`] the bounding box used to crop the image in self
            shape: [tuple] (height, width) of the uncropped image
    
        Returns:
            this `vipy.image.Image` object with the pixels uncropped.

        .. note:: NOT idempotent.  This will generate different results if run more than once.
        """
        ((x,y,w,h), (H,W)) = (bb.xywh(), shape)
        ((dyb, dya), (dxb, dxa)) = ((int(y), int(H-(y+h))), (int(x), int(W-(x+w))))
        self._array = np.pad(self.load().array(),
                             ((dyb, dya), (dxb, dxa), (0, 0)) if
                             self.load().array().ndim == 3 else ((dyb, dya), (dxb, dxa)),
                             mode='constant')        
        return self

    def splat(self, im, bb):
        """Replace pixels within boundingbox in self with pixels in im"""
        assert isinstance(im, vipy.image.Image), "invalid image"
        assert (im.width() == bb.width() and im.height() == bb.height()) or bb.isinterior(im.width(), im.height()) and bb.isinterior(self.width(), self.height()), "Invalid bounding box '%s'" % str(bb)
        (x,y,w,h) = bb.xywh()
        self._array[int(y):int(y+h), int(x):int(x+w)] = im.array() if (im.width() == bb.width() and im.height() == bb.height()) else im.array()[int(y):int(y+h), int(x):int(x+w)]
        return self            
        
    def store(self):
        """Store the current image file as an attribute of this object.  Useful for archiving an object to be fully self contained without any external references.  
        
           -Remove this stored image using unstore()
           -Unpack this stored image and set up the filename using restore() 
           -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string.
           -Useful for creating a single self contained object for distributed processing.  

        ```python
        v == v.store().restore(v.filename()) 
        ```

        """
        assert self.hasfilename(), "Image file not found"
        with open(self.filename(), 'rb') as f:
            self.attributes['__image__'] = f.read()
        return self

    def unstore(self):
        """Delete the currently stored image from store()"""
        return self.delattribute('__image__')

    def restore(self, filename):
        """Save the currently stored image to filename, and set up filename"""
        assert self.hasattribute('__image__'), "Image not stored"
        with open(filename, 'wb') as f:
            f.write(self.attributes['__image__'])
        return self.filename(filename)                
    
    def abspath(self):
        """Change the path of the filename from a relative path to an absolute path (not relocatable)"""
        return self.filename(os.path.normpath(os.path.abspath(os.path.expanduser(self.filename()))))

    def relpath(self, parent=None):
        """Replace the filename with a relative path to parent (or current working directory if none)"""
        parent = parent if parent is not None else os.getcwd()
        assert parent in os.path.expanduser(self.filename()), "Parent path '%s' not found in abspath '%s'" % (parent, self.filename())
        return self.filename(PurePath(os.path.expanduser(self.filename())).relative_to(parent))

    def canload(self):
        """Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images"""
        if not self.isloaded():
            try:
                if isimagefile(self._filename) and os.path.exists(self._filename):
                    PIL.Image.open(self._filename).verify()  # faster, throws exception on corrupted image
                else:
                    self.load().flush()  # fallback, load it and flush to avoid memory leak (expensive)
                return True
            except:
                return False
        else:
            return True
        
    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(s=None, encode=False)

    def json(self, s=None, encode=True):
        if s is None:
            d = {'filename':self._filename,
                 'url':self._url,
                 'loader':self._loader,
                 'array':self._array.tolist() if self._array is not None else None,
                 'colorspace':self._colorspace,
                 'attributes':self.attributes}                        
            return json.dumps(d) if encode else d
        else:
            d = json.loads(s)
            d = {k.lstrip('_'):v for (k,v) in d.items()}  # prettyjson (remove "_" prefix to attributes)                            
            self._filename = d['filename']
            self._url = d['url']
            self._loader = d['loader']
            self._array = np.array(d['array'], dtype=np.uint8) if d['array'] is not None else None
            self._colorspace = d['colorspace']
            self.attributes = d['attributes']            
            return self
        
    def loader(self, f):
        """Lambda function to load an unsupported image filename to a numpy array.
        
        This lambda function will be executed during load and the result will be stored in self._array
        """
        self._loader = f
        return self

    def has_loader(self):
        return self._loader is not None
    
    def load(self, ignoreErrors=False, verbose=True):
        """Load image to cached private '_array' attribute.

        Args:
            ignoreErrors: [bool] If true, ignore any exceptions thrown during load and print the corresponding error messages.  This is useful for loading images distributed without throwing exceptions when some images may be corrupted.  In this case, the _array attribute will be None and `vipy.image.Image.isloaded` will return false to determine if the image is loaded, which can be used to filter out corrupted images gracefully. 
            verbose: [bool] If true, show additional useful printed output

        Returns:
            This `vipy.image.Image` object with the pixels loaded in self._array as a numpy array.

        .. note:: This loader supports any image file format supported by PIL.  A custom loader can be added using `vipy.image.Image.loader`.
        """
        try:
            # Return if previously loaded image
            if self._array is not None:
                return self

            # Download URL to filename if cached filename does not exist
            if self._url is not None and not self.hasfilename():
                if self._filename is None and os.path.exists(os.path.join(str(cache()), filetail(self._url))):
                    self._filename = os.path.join(str(cache()), filetail(self._url))
                else:
                    self.download(ignoreErrors=ignoreErrors, verbose=verbose)

            # Load filename to numpy array
            if self._loader is not None:
                self._array = self._loader(self._filename)  
                if self.isluminance():
                    self.colorspace('lum')
                elif self.iscolor():
                    self.colorspace('rgb')
                else:
                    self._array = np.float32(self._array)
                    self.colorspace('float') 
            elif isimagefile(self._filename):
                self._array = np.array(PIL.Image.open(self._filename))  # RGB order!
                if self.istransparent():
                    self.colorspace('rgba')  # must be before iscolor()
                elif self.iscolor():
                    self.colorspace('rgb')
                elif self.isgrey():
                    self.colorspace('grey')
                elif self.isluminance():
                    self.colorspace('lum')
                else:
                    log.warning('unknown colorspace for image "%s" - attempting to coerce to colorspace=float' % str(self._filename))
                    self._array = np.float32(self._array)
                    self.colorspace('float')
            elif iswebp(self._filename):
                import vipy.video
                return vipy.video.Video(self._filename).load()  
            elif hasextension(self._filename):
                raise ValueError('Non-standard image extensions require a custom loader')
            else:
                # Attempting to open it anyway, may be an image file without an extension. Cross your fingers ...
                self._array = np.array(PIL.Image.open(self._filename))  # RGB order!

        except IOError:
            if ignoreErrors:
                if verbose is True:
                    log.warning('[vipy.image]: IO error "%s" -> "%s" ' % (self.url(), self.filename()))
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if ignoreErrors:
                if verbose is True:
                    log.warning('[vipy.image]: Load error for image "%s"' % self.filename())
                self._array = None
            else:
                raise

        return self

    def download(self, ignoreErrors=False, timeout=10, verbose=True):
        """Download URL to filename provided by constructor, or to temp filename.

        Args:
            ignoreErrors:  [bool] If true, do not throw an exception if the download of the URL fails for some reason.  Instead, print out a reason and return this image object.  The function `vipy.image.Image.hasfilename` will return false if the downloaded file does not exist and can be used to filter these failed downloads gracefully.
            timeout: [int]  The timeout in seconds for an http or https connection attempt.  See also [urllib.request.urlopen](https://docs.python.org/3/library/urllib.request.html).
            verbose: [bool] If true, output more helpful message.

        Returns:
            This `vipy.image.Image` object with the URL downloaded to `vipy.image.Image.filename` or to a `vipy.util.tempimage` filename which can be retrieved with `vipy.image.Image.filename`.
        """
        if self._url is None and self._filename is not None:
            return self
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[vipy.image.download][ERROR]: '
                             'Invalid URL "%s" ' % self._url)

        if self._filename is None:
            if vipy.globals.cache() is not None:
                self._filename = os.path.join(remkdir(vipy.globals.cache()), filetail(self._url.split('?')[0]))
                self._filename = self._filename+'.jpg' if not hasextension(self._filename) else self._filename  # guess JPG for URLs with no file extension  
            elif isimageurl(self._url):
                self._filename = tempimage(fileext(self._url))
            else:
                self._filename = tempjpg()  # guess JPG for URLs with no file extension

        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if url_scheme in ['http', 'https']:
                vipy.downloader.download(self._url,
                                         self._filename,
                                         verbose=verbose,
                                         timeout=timeout,
                                         sha1=self._urlsha1 if hasattr(self, '_urlsha1') else None,
                                         username=self._urluser if hasattr(self, '_urluser') else None,
                                         password=self._urlpassword if hasattr(self, '_urlpassword') else None)
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
                if verbose is True:
                    log.warning('[vipy.image]: download failed for url "%s"' % self._url)
                self._array = None
            else:
                raise

        except IOError:
            if ignoreErrors:
                if verbose:
                    log.warning('[vipy.image]: IO error downloading "%s" -> "%s" ' % (self.url(), self.filename()))
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if ignoreErrors:
                if verbose:
                    log.warning('[vipy.image]: load error for image "%s"' % self.filename())
            else:
                raise

        return self

    def reload(self):
        """Flush the image buffer to force reloading from file or URL"""
        return self.clone(flush=True).load()

    def isloaded(self):
        """Return True if `vipy.image.Image.load` was successful in reading the image, or if the pixels are present in `vipy.image.Image.array`."""
        return self._array is not None

    def isdownloaded(self):
        """Does the filename returned from `vipy.image.Image.filename` exist, meaning that the url has been downloaded to a local file?"""
        return self._filename is not None and os.path.exists(self._filename)
    
    def downloadif(self, ignoreErrors=False, timeout=10, verbose=True):
        """Download URL to filename if the filename has not already been downloaded"""
        return self.download(ignoreErrors=ignoreErrors, timeout=timeout, verbose=verbose) if self.hasurl() and not self.isdownloaded() else self
    
    def channels(self):
        """Return integer number of color channels"""
        return 1 if self.load().array().ndim == 2 else self.load().array().shape[2]

    def iscolor(self):
        """Color images are three channel or four channel with transparency, float32 or uint8"""
        return self.channels() == 3 or self.channels() == 4

    def istransparent(self):
        """Transparent images are four channel color images with transparency, float32 or uint8.  Return true if this image contains an alpha transparency channel"""
        return self.channels() == 4

    def isgrey(self):
        """Grey images are one channel, float32"""
        return self.channels() == 1 and self.array().dtype == np.float32

    def isluminance(self):
        """Luninance images are one channel, uint8"""
        return self.channels() == 1 and self.array().dtype == np.uint8

    def filesize(self):
        """Return size of underlying image file, requires fetching metadata from filesystem"""
        assert self.hasfilename(), 'Invalid image filename'
        return os.path.getsize(self._filename)

    def width(self):
        """Return the width (columns) of the image in integer pixels.
        
        .. note:: This triggers a `vipy.image.Image.load` if the image is not already loaded.
        """
        return self.load().array().shape[1]

    def height(self):
        """Return the height (rows) of the image in integer pixels.
        
        .. note:: This triggers a `vipy.image.Image.load` if the image is not already loaded.
        """        
        return self.load().array().shape[0]
    
    def shape(self):
        """Return the (height, width) or equivalently (rows, cols) of the image.
        
        Returns:
            A tuple (height=int, width=int) of the image.

        .. note:: This triggers a `vipy.image.Image.load` if the image is not already loaded.
        """
        return (self.load().height(), self.width())

    def aspectratio(self):
        """Return the aspect ratio of the image as (width/height) ratio.

        Returns:
            A float equivalent to (`vipy.image.Image.width` / `vipy.image.Image.height`)

        .. note:: This triggers a `vipy.image.Image.load` if the image is not already loaded.
        """
        return self.load().width() / float(self.height())

    def area(self):
        """Return the area of the image as (width * height).

        Returns:
            An integer equivalent to (`vipy.image.Image.width` * `vipy.image.Image.height`)

        .. note:: This triggers a `vipy.image.Image.load` if the image is not already loaded.
        """
        return self.width()*self.height()
    
    def centroid(self):
        """Return the real valued center pixel coordinates of the image (col=x,row=y).
        
        The centroid is equivalent to half the `vipy.image.Image.shape`.

        Returns:
            A tuple (column, row) of the floating point center of the image.
        """
        return (self.load().width() / 2.0, self.height() / 2.0)

    def centerpixel(self):
        """Return the integer valued center pixel coordinates of the image (col=i,row=j)

        The centerpixel is equivalent to half the `vipy.image.Image.shape` floored to the nearest integer pixel coordinate.

        Returns:
            A tuple (int(column), int(row)) of the integer center of the image.
        """
        c = np.round(self.centroid())
        return (int(c[0]), int(c[1]))
    
    def array(self, np_array=None, copy=False):
        """Replace self._array with provided numpy array

        Args:
            np_array: [numpy array] A new array to use as the pixel buffer for this image.
            copy: [bool] If true, copy the buffer using np.copy(), else use a reference to this buffer.

        Returns:
            - If np_array is not None, return the `vipy.image.Image` object such that this object points to the provided numpy array as the pixel buffer
            - If np_array is None, then return the numpy array.

        .. notes:: 
            - If copy=False, then this `vipy.image.Image` object will share the pixel buffer with the owner of np_array.  Changes to pixels in this buffer will be shared.  
            - If copy=True, then this will significantly slow down processing for large images.  Use referneces wherevery possible.
        """
        if np_array is None:
            return self._array if copy is False else np.copy(self._array)
        elif isnumpyarray(np_array):
            self._array = np.copy(np_array) if copy else np_array  # reference or copy
            assert self._array.dtype == np.float32 or self._array.dtype == np.uint8, "Invalid input - array() must be type uint8 or float32 and not type='%s'" % (str(self._array.dtype))                        
            self.colorspace(None)  # must be set with colorspace() after array() but before _convert()
            return self
        else:
            raise ValueError('Invalid input - array() must be numpy array and not "%s"' % (str(type(np_array))))

    def fromarray(self, data):
        """Alias for `vipy.image.Image.array` with copy=True. This will set new numpy array as the pixel buffer with a numpy array copy"""
        return self.array(data, copy=True)
    
    def tonumpy(self):
        """Alias for `vipy.image.Image.numpy"""
        return self.numpy()

    def numpy(self):
        """Return a mutable numpy array for this `vipy.image.Image`.

        .. notes:: 
            - This will always return a writeable array with the 'WRITEABLE' numpy flag set.  This is useful for returning a mutable numpy array as needed while keeping the original non-mutable numpy array (e.g. loaded from a video or PIL) as the underlying pixel buffer for efficiency reasons.
            - Triggers a `vipy.image.Image.load` if the pixel buffer has not been loaded
            - This will trigger a copy if the ['WRITEABLE' flag](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) is not set. 
        """        
        self.load()
        self._array = np.copy(self._array) if not self._array.flags['WRITEABLE'] else self._array  # triggers copy         
        return self._array
        
    def channel(self, k=None):
        """Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None.

        Iterate over channels as single channel luminance images:

        ```python
        for c in self.channel():
            print(c)
        ```

        Return the kth channel as a single channel luminance image:

        ```python
        c = self.channel(k=0)
        ```

        """
        if k is None:
            return [self.channel(j) for j in range(0, self.channels())]
        elif k == 0 and self.channels() == 1:
            return self
        else:
            assert k < self.channels() and k>=0, "Requested channel=%d must be within valid channels=%d" % (k, self.channels())
            im = self.clone().load()
            im._array = im._array[:,:,k]
            im._colorspace = 'lum'
            return im

    def channelmean(self):
        """Return a cloned Image() object for the mean of all channels followed by returning a single channel float image.

        This is useful for visualizing multichannel images by reducing the channels to one

        ```python
        vipy.image.Image(array=np.random.rand(3,3,16).astype(np.float32)).channelmean().mat2gray().lum().show()
        ```
        
        """
        im = self.clone().load()
        im._array = np.mean(im._array, axis=2, keepdims=True)
        im._colorspace = 'float'
        return im
        
    def red(self):
        """Return red channel as a cloned single channel `vipy.image.Image` object.

        These are equivalent operations if the colorspace is 'rgb' or 'rgba':
        
        ```python
        self.red() == self.channel(0) 
        ```

        These are equivalent operations if the colorspace is 'bgr' or 'bgra':

        ```python
        self.red() == self.channel(3) 
        ```

        .. note:: OpenCV returns images in BGR colorspace.  Use this method to always return the desired channel by color.
        """
        assert self.channels() >= 3, "Must be color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(0)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(3)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())

    def green(self):
        """Return green channel as a cloned single channel `vipy.image.Image` object.

        These are equivalent operations if the colorspace is 'rgb' or 'rgba':

        ```python
        self.green() == self.channel(1) 
        ```

        These are equivalent operations if the colorspace is 'bgr' or 'bgra':

        ```python
        self.green() == self.channel(1) 
        ```

        .. note:: OpenCV returns images in BGR colorspace.  Use this method to always return the desired channel by color.
        """
        assert self.channels() >= 3, "Must be three channel color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(1)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(1)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())

    def blue(self):
        """Return blue channel as a cloned single channel `vipy.image.Image` object.

        These are equivalent operations if the colorspace is 'rgb' or 'rgba':

        ```python
        self.vlue() == self.channel(2) 
        ```

        These are equivalent operations if the colorspace is 'bgr' or 'bgra':

        ```python
        self.blue() == self.channel(0) 
        ```

        .. note:: OpenCV returns images in BGR colorspace.  Use this method to always return the desired channel by color.
        """
        assert self.channels() >= 3, "Must be three channel color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(2)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(0)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())                

    def alpha(self):
        """Return alpha (transparency) channel as a cloned single channel `vipy.image.Image` object"""
        assert self.channels() == 4 and self.colorspace() in ['rgba', 'bgra'], "Must be four channnel color image"
        return self.channel(3)
        
    def zeros(self):
        """Set the pixel buffer to all zeros of the same shape and datatype as this `vipy.image.Image` object.
        
        These are equivalent operations for the resulting buffer shape: 
        
        ```python
        import numpy as np
        np.zeros( (self.width(), self.height(), self.channels()) ) == self.zeros().array()
        ```

        Returns:
           This `vipy.image.Image` object.

        .. note:: Triggers load() if the pixel buffer has not been loaded yet.
        """
        self._array = 0*self.load()._array
        return self

    def pil(self):
        """Convert vipy.image.Image to PIL Image.
        
        Returns:
            A [PIL image](https://pillow.readthedocs.io/en/stable/reference/Image.html) object, that shares the pixel buffer by reference
        """
        if self.isloaded():
            assert self.channels() in [1,3,4] and (self.channels() == 1 or self.colorspace() != 'float'), "Incompatible with PIL"
            return PIL.Image.fromarray(self.numpy(), mode='RGB' if self.colorspace()=='rgb' else None)  # FIXME: mode='RGB' triggers slow tobytes() conversion, need RGBA or RGBX
        elif self.hasfilename():
            return PIL.Image.open(self.filename())
        else:
            return None
            
    def blur(self, sigma=3):
        """Apply a Gaussian blur with Gaussian kernel radius=sigma to the pixel buffer.
        
        Args:
            sigma: [float >=0] The gaussian blur kernel radius.

        Returns:
            This `vipy.image.Image` object with the pixel buffer blurred in place.
        """
        assert sigma >= 0
        return self.array(np.array(self.pil().filter(PIL.ImageFilter.GaussianBlur(radius=sigma)))) if sigma>0 else self
        
    def torch(self, order='CHW'):
        """Convert the batch of 1 HxWxC images to a CxHxW torch tensor.

        Args:
            order: ['CHW', 'HWC', 'NCHW', 'NHWC'].  The axis order of the torch tensor (channels, height, width) or (height, width, channels) or (1, channels, height, width) or (1, height, width, channels)

        Returns:
            A CxHxW or HxWxC or 1xCxHxW or 1xHxWxC [torch tensor](https://pytorch.org/docs/stable/tensors.html) that shares the pixel buffer of this image object by reference.

        .. note:: This supports numpy types and does not support bfloat16
        """
        assert order in ['CHW', 'HWC', 'NCHW', 'NHWC']
        img = self.array() if self.array().ndim == 3 else np.expand_dims(self.array(), 2)  # HxW -> HxWx1, HxWxC -> HxWxC (unchanged)
        img = img.transpose(2,0,1) if order in ['CHW', 'NCHW']  else img   # HxWxC -> CxHxW
        img = np.expand_dims(img,0) if order in ['NHWC', 'NCHW'] else img  # HxWxC -> 1xHxWxC or CxHxW -> 1xCxHxW
        return torch.from_numpy(img)   # pip install torch

    @staticmethod
    def fromtorch(x, order='CHW'):
        """Convert a 1xCxHxW or CxHxW torch tensor (or numpy array with torch channel order) to HxWxC numpy array, returns new `vipy.image.Image` with inferred colorspace corresponding to data type in x"""
        try_import('torch'); import torch        
        assert isinstance(x, torch.Tensor) or isinstance(x, np.ndarray), "Invalid input type '%s'- must be torch.Tensor" % (str(type(x)))
        assert (x.ndim == 4 and x.shape[0] == 1) or x.ndim == 3, "Torch tensor must be shape 1xCxHxW or CxHxW"
        x = x.squeeze(0) if (x.ndim == 4 and x.shape[0] == 1) else x

        if order == 'CHW':
            x = x.permute(1,2,0).cpu().detach().float().numpy() if torch.is_tensor(x) else x.transpose(1,2,0)   # CxHxW -> HxWxC, copied            
        elif order == 'WHC':
            x = x.permute(1,0,2).cpu().detach().float().numpy() if torch.is_tensor(x) else x.transpose(1,0,2)   # WxHxC -> HxWxC, copied        
        elif order == 'HWC':
            x = x.cpu().detach().float().numpy() if torch.is_tensor(x) else x  # HxWxC -> HxWxC, copied        
        else:
            raise ValueError('unknown axis order "%s"' % order)

        img = np.copy(x)
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace  # assumed
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace
        return Image(array=img, colorspace=colorspace)
    
    def nofilename(self):
        self._filename = None
        return self

    def unload(self):
        """Remove cached file and loaded array.  Note that this will delete the underlying file returned by filename() if there is a backing url"""
        if self.hasurl() and self.hasfilename():
            os.remove(self._filename)
            self._filename = None
        if self.isloaded():
            self.flush()
        return self
    
    def filename(self, newfile=None):
        """Return or set image filename"""
        if newfile is None:
            return self._filename
        else:
            self._filename = newfile
            return self

    def nourl(self):
        self._url = None
        return self

    def url(self, url=None, username=None, password=None, sha1=None):
        """Image URL and URL download properties"""
        if url is not None:
            self._url = url  # this does not change anything else (e.g. the associated filename), better to use constructor 
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
    
    def colorspace(self, colorspace=None):
        """Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']"""
        if colorspace is None:
            return self._colorspace
        else:
            assert str(colorspace).lower() in ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'gray', 'lum'], "Invalid colorspace '%s'. Allowable is ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'gray', 'lum']" % colorspace 
            img = self.array()
            if self.isloaded():
                colorspace = str(colorspace).lower()
                if self.array().dtype == np.float32:
                    assert colorspace in ['float', 'grey', 'gray'], "Invalid colorspace '%s' for float32 array()" % colorspace
                elif self.array().dtype == np.uint8:
                    assert colorspace in  ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'lum'], "Invalid colorspace '%s' for uint8 array(). Allowable is ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'lum']" % colorspace
                else:
                    raise ValueError('unupported array() datatype "%s".  Allowable is [np.float32, np.uint8]' % colorspace)  # should never get here as long as array() is used to set _array
                if self.channels() == 1:
                    assert colorspace in ['float', 'grey', 'gray', 'lum'], "Invalid colorspace '%s; for single channel array.  Allowable is ['float', 'grey', 'gray', 'lum']" % colorspace
                elif self.channels() == 3:
                    assert colorspace in ['float', 'rgb', 'bgr', 'hsv'], "Invalid colorspace '%s; for three channel array. Allowable is ['float', 'rgb', 'bgr', 'hsv']" % colorspace
                elif self.channels() == 4:
                    assert colorspace in ['float', 'rgba', 'bgra'], "Invalid colorspace '%s; for four channel array. Allowable is ['float', 'rgba', 'bgra']" % colorspace                    
                elif colorspace != 'float':
                    raise ValueError("Invalid colorspace '%s' for image channels=%d, type=%s" % (colorspace, self.channels(), str(self.array().dtype)))
                if colorspace in ['grey', 'gray']:
                    assert self.max() <= 1 and self.min() >= 0, "Colorspace 'grey' image must be np.float32 in range [0,1].  Use colorspace 'lum' for np.uint8 in range [0,255], or colorspace 'float' for unconstrained np.float32 [-inf, +inf]"
                    colorspace = 'grey'  # standardize
            self._colorspace = str(colorspace).lower()
            return self

    def uri(self):
        """Return the URI of the image object, either the URL or the filename, raise exception if neither defined"""
        if self.hasurl():
            return self.url()
        elif self.hasfilename():
            return self.filename()
        else:
            raise ValueError('No URI defined')

    def setattribute(self, key, value):
        """Set element self.attributes[key]=value"""
        if self.attributes is None:
            self.attributes = {key: value}
        else:
            self.attributes[key] = value
        return self

    def setattributes(self, newattr):
        """Set many attributes at once by providing a dictionary to be merged with current attributes"""
        assert isinstance(newattr, dict), "New attributes must be dictionary"
        self.attributes.update(newattr)
        return self
    
    def getattribute(self, k):
        """Return the key k in the attributes dictionary (self.attributes) if present, else None"""        
        return self.attributes[k] if k in self.attributes else None        

    def hasattribute(self, key):
        return self.attributes is not None and key in self.attributes

    def delattribute(self, k):
        if k in self.attributes:
            self.attributes.pop(k)
        return self

    def delattributes(self, atts):
        for k in tolist(atts):
            self.delattribute(k)
        return self
    
    def metadata(self, k=None):
        """Return metadata associated with this image, stored in the attributes dictionary"""
        return self.attributes if k is None else self.getattribute(k)
    
    def hasurl(self):
        """synonym for `vipy.image.has_url`"""
        return self.has_url()

    def has_url(self):
        """Return True if the image has a URL input source"""
        return self._url is not None and isurl(self._url)
    
    def hasfilename(self):
        return self._filename is not None and os.path.exists(self._filename)

    def clone(self, flushforward=False, flushbackward=False, flush=False, shallow=False, attributes=False):
        """Create deep copy of object, flushing the original buffer if requested and returning the cloned object.
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
        elif shallow:
            im = copy.copy(self)  # shallow copy
            im._array = np.asarray(self._array) if self._array is not None else None  # shared pixels            
        else:
            im = copy.deepcopy(self)
        if attributes:
            im.attributes = copy.deepcopy(self.attributes)
        return im

    def flush(self):
        """Alias for clone(flush=True), returns self not clone"""
        self._array = None  # flushes buffer on object and clone
        return self

        
    # Spatial transformations
    def resize(self, cols=None, rows=None, width=None, height=None, interp='bilinear', fast=False):
        """Resize the image buffer to (rows x cols) with bilinear interpolation.  If rows or cols is provided, rescale image maintaining aspect ratio"""
        assert not (cols is not None and width is not None), "Define either width or cols"
        assert not (rows is not None and height is not None), "Define either height or rows"
        rows = rows if height is None else height
        cols = cols if width is None else width
        if cols is None or rows is None:
            if cols is None:
                scale = float(rows) / float(self.height())
            else:
                scale = float(cols) / float(self.width())
            self.rescale(scale)
        elif rows == self.height() and cols == self.width():
            return self  
        elif self.colorspace() == 'float':
            self._array = np.dstack([np.array(im.pil().resize((cols, rows), string_to_pil_interpolation(interp))) for im in self.channel()])
        else:
            self._array = np.asarray(self.load().pil().resize((cols, rows), string_to_pil_interpolation(interp), reducing_gap=2 if fast else None))  
        return self

    def resize_like(self, im, interp='bilinear'):
        """Resize image buffer to be the same size as the provided vipy.image.Image()"""
        assert isinstance(im, Image), "Invalid input - Must be vipy.image.Image"
        return self.resize(im.width(), im.height(), interp=interp)
    
    def rescale(self, scale=1, interp='bilinear', fast=False):
        """Scale the image buffer by the given factor - NOT idempotent"""
        (height, width) = self.load().shape()
        if scale == 1:
            return self
        elif self.colorspace() == 'float':
            self._array = np.dstack([np.asarray(im.pil().resize((int(np.round(scale * width)), int(np.round(scale * height))), string_to_pil_interpolation(interp))) for im in self.channel()])
        else: 
            self._array = np.asarray(self.pil().resize((int(np.round(scale * width)), int(np.round(scale * height))), string_to_pil_interpolation(interp), reducing_gap=2 if fast else None))
        return self

    def maxdim(self, dim=None, interp='bilinear'):
        """Resize image preserving aspect ratio so that maximum dimension of image = dim, or return maxdim()"""
        return self.rescale(float(dim) / float(np.maximum(self.height(), self.width())), interp=interp) if dim is not None else max(self.shape())

    def mindim(self, dim=None, interp='bilinear'):
        """Resize image preserving aspect ratio so that minimum dimension of image = dim, or return mindim()"""
        if dim is None:
            return np.minimum(self.height(), self.width())
        else:
            s = float(dim) / float(np.minimum(self.height(), self.width()))
            return self.rescale(s, interp=interp) if dim is not None else min(self.shape())

    def mindimn(self, dim=None):
        """Frequently used shortcut for mindim(dim, interp='nearest')"""
        return self.mindim(dim, interp='nearest')
    
    def _pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode, dx=padwidth, dy=padheight, thin wrapper for numpy.pad"""
        self._array = np.pad(self.load().array(),
                             ((dy, dy), (dx, dx), (0, 0)) if
                             self.load().array().ndim == 3 else ((dy, dy), (dx, dx)),
                             mode=mode)
        return self

    def zeropad(self, padwidth, padheight):
        """Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding,, and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding"""
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth, padwidth)
        if not isinstance(padheight, tuple):
            padheight = (padheight, padheight)
        if self.iscolor():
            pad_shape = (padheight, padwidth, (0, 0))
        else:
            pad_shape = (padheight, padwidth)

        assert all([x>=0 for x in padheight]) and all([x>=0 for x in padwidth]), "padding must be positive"
        if padwidth[0]>0 or padwidth[1]>0 or padheight[0]>0 or padheight[1]>0:
            self._array = np.pad(self.load().array(), pad_width=pad_shape, mode='constant', constant_values=0)  # this is still slow due to the required copy, but fast-ish in np >= 1.17
            
        return self

    def zeropadlike(self, width, height):
        """Zero pad the image balancing the border so that the resulting image size is (width, height)"""
        assert width >= self.width() and height >= self.height(), "Invalid input - final (width=%d, height=%d) must be greater than current image size (width=%d, height=%d)" % (width, height, self.width(), self.height())
        return self.zeropad( (int(np.floor((width - self.width())/2)), int(np.ceil((width - self.width())/2))),
                             (int(np.floor((height - self.height())/2)), int(np.ceil((height - self.height())/2))))
                            
    def meanpad(self, padwidth, padheight, mu=None):
        """Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding,, and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding"""        
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth, padwidth)
        if not isinstance(padheight, tuple):
            padheight = (padheight, padheight)
        assert all([x>=0 for x in padheight]) and all([x>=0 for x in padwidth]), "padding must be positive"
        mu = self.meanchannel() if mu is None else mu
        self._array = np.squeeze(np.dstack([np.pad(img,
                                                   pad_width=(padheight,padwidth),
                                                   mode='constant',
                                                   constant_values=c) for (img,c) in zip(self.channel(), mu)]))
        return self

    def alphapad(self, padwidth, padheight):
        """Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding,, and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding"""
        assert self.colorspace() == 'rgba', "Colorspace must be RGBA for padding with transparency"
        return self.meanpad(padwidth, padheight, mu=np.array([0,0,0,0]))
    
    def minsquare(self):
        """Crop image of size (HxW) to (min(H,W), min(H,W)), keeping upper left corner constant"""
        S = np.min(self.load().shape())
        return self._crop(BoundingBox(xmin=0, ymin=0, width=int(S), height=int(S)))

    def maxsquare(self, S=None):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding or (S,S) if provided, keeping upper left corner constant"""
        S = np.max(self.load().shape()) if S is None else int(S)
        (H, W) = self.shape()
        (dW, dH) = (max(0, S - W), max(0, S - H))
        if S != W or S != H:
            self._crop(BoundingBox(0, 0, width=min(W, S), height=min(H, S)))
            if (dW > 0 or dH > 0):
                self.zeropad((0,dW), (0,dH))  # crop then zeropad
        return self

    def maxmatte(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte"""
        S = np.max(self.load().shape())
        dW = S - self.width()
        dH = S - self.height()
        return self.zeropad((int(np.floor(dW//2)), int(np.ceil(dW//2))), (int(np.floor(dH//2)), int(np.ceil(dH//2))))._crop(BoundingBox(0, 0, width=int(S), height=int(S)))
    
    def centersquare(self):
        """Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant"""
        N = int(np.min(self.shape()))
        return self._crop(BoundingBox(xcentroid=float(self.width() / 2.0), ycentroid=float(self.height() / 2.0), width=N, height=N))

    def centercrop(self, height, width):
        """Crop image of size (height x width) in the center, keeping the image centroid constant"""
        return self._crop(BoundingBox(xcentroid=float(self.width() / 2.0), ycentroid=float(self.height() / 2.0), width=int(width), height=int(height)))

    def cornercrop(self, height, width):
        """Crop image of size (height x width) from the upper left corner"""
        return self._crop(BoundingBox(xmin=0, ymin=0, width=int(width), height=int(height)))
    
    def _crop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, clipping the box to the image rectangle"""
        assert isinstance(bbox, BoundingBox) and bbox.valid(), "Invalid input - Must be vipy.geometry.BoundingBox not '%s'" % (str(type(bbox)))
        if not bbox.isdegenerate() and bbox.hasoverlap(self.load().array()):
            bbox = bbox.imclip(self.load().array()).int()
            self._array = self.array()[bbox.ymin():bbox.ymax(),
                                       bbox.xmin():bbox.xmax()]
        else:
            log.warning('BoundingBox for crop() does not intersect image rectangle')
        return self

    def crop(self, bbox):
        return self._crop(bbox)
    
    def fliplr(self):
        """Mirror the image buffer about the vertical axis - Not idempotent"""
        self._array = np.fliplr(self.load().array())
        return self

    def flipud(self):
        """Mirror the image buffer about the horizontal axis - Not idempotent"""
        self._array = np.flipud(self.load().array())
        return self
    
    def imagebox(self):
        """Return the bounding box for the image rectangle"""
        return BoundingBox(xmin=0, ymin=0, width=int(self.width()), height=int(self.height()))

    def border_mask(self, pad):
        """Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge"""
        img = np.zeros( (self.height(), self.width()), dtype=np.uint8)
        img[0:pad,:] = 1
        img[-pad:,:] = 1
        img[:,0:pad] = 1
        img[:,-pad:] = 1
        return img
    
    # Color conversion
    def _convert(self, to):
        """Supported colorspaces are rgb, rgbab, bgr, bgra, hsv, grey, lum, float"""
        to = to if to != 'gray' else 'grey'  # standardize 'gray' -> 'grey' internally
        self.load()
        if self.colorspace() == to:
            return self
        elif to == 'float':
            img = self.load().array()  # any type
            self._array = np.array(img).astype(np.float32)  # typecast to float32
        elif self.colorspace() == 'lum':
            img = self.load().array()  # single channel, uint8 [0,255]
            assert img.dtype == np.uint8
            img = np.squeeze(img, axis=2) if img.ndim == 3 and img.shape[2] == 1 else img  # remove singleton channel            
            self._array = np.array(PIL.Image.fromarray(img, mode='L').convert('RGB'))  # uint8 luminance [0,255] -> uint8 RGB
            self.colorspace('rgb')
            self._convert(to)
        elif self.colorspace() in ['gray', 'grey']:
            img = self.load().array()  # single channel float32 [0,1]
            img = np.squeeze(img, axis=2) if img.ndim == 3 and img.shape[2] == 1 else img  # remove singleton channel                        
            self._array = np.array(PIL.Image.fromarray(255.0 * img, mode='F').convert('RGB'))  # float32 gray [0,1] -> float32 gray [0,255] -> uint8 RGB
            self.colorspace('rgb')
            self._convert(to)
        elif self.colorspace() == 'rgba':
            img = self.load().array()  # uint8 RGBA
            if to == 'bgra':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGBA -> uint8 ABGR
                self._array = self._array[:,:,[1,2,3,0]]  # uint8 ABGR -> uint8 BGRA
            elif to == 'rgb':
                self._array = self._array[:,:,0:-1]  # uint8 RGBA -> uint8 RGB
            else:
                self._array = self._array[:,:,0:-1]  # uint8 RGBA -> uint8 RGB
                self.colorspace('rgb')
                self._convert(to)
        elif self.colorspace() == 'rgb':
            img = self.load().array()  # uint8 RGB
            if to in ['grey', 'gray']:
                self._array = (1.0 / 255.0) * np.array(PIL.Image.fromarray(img).convert('L')).astype(np.float32)  # uint8 RGB -> float32 Grey [0,255] -> float32 Grey [0,1]
            elif to == 'bgr':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGB -> uint8 BGR
            elif to == 'hsv':
                self._array = np.array(PIL.Image.fromarray(img).convert('HSV'))  # uint8 RGB -> uint8 HSV
            elif to == 'lum':
                self._array = np.array(PIL.Image.fromarray(img).convert('L'))  # uint8 RGB -> uint8 Luminance (integer grey)
            elif to == 'rgba':
                self._array = np.dstack((img, 255*np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)))
            elif to == 'bgra':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGB -> uint8 BGR
                self._array = np.dstack((self._array, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))  # uint8 BGR -> uint8 BGRA
        elif self.colorspace() == 'bgr':
            img = self.load().array()  # uint8 BGR
            self._array = np.array(img)[:,:,::-1]  # uint8 BGR -> uint8 RGB
            self.colorspace('rgb')
            self._convert(to)
        elif self.colorspace() == 'bgra':
            img = self.load().array()  # uint8 BGRA
            self._array = np.array(img)[:,:,::-1]  # uint8 BGRA -> uint8 ARGB
            self._array = self._array[:,:,[1,2,3,0]]  # uint8 ARGB -> uint8 RGBA
            self.colorspace('rgba')
            self._convert(to)
        elif self.colorspace() == 'hsv':
            img = self.load().array()  # uint8 HSV
            self._array = np.array(PIL.Image.fromarray(img, mode='HSV').convert('RGB'))  # uint8 HSV -> uint8 RGB
            self.colorspace('rgb')
            self._convert(to)
        elif self.colorspace() == 'float':
            img = self.load().array()  # float32
            if np.max(img) > 1 or np.min(img) < 0:
                log.warning('Converting float image to "%s" will be rescaled with self.mat2gray() into the range float32 [0,1]' % to)
                img = self.mat2gray().array()
            if not self.channels() in [1,2,3]:
                raise ValueError('Float image must be single channel or three channel RGB in the range float32 [0,1] prior to conversion')
            if self.channels() == 3:  # assumed RGB
                self._array = np.uint8(255 * self.array())   # float32 RGB [0,1] -> uint8 RGB [0,255]
                self.colorspace('rgb')
            else:
                img = np.squeeze(img, axis=2) if img.ndim == 3 else img
                self._array = (1.0 / 255.0) * np.array(PIL.Image.fromarray(np.uint8(255 * img)).convert('L')).astype(np.float32)  # float32 RGB [0,1] -> float32 gray [0,1]                
                self.colorspace('grey')
            self._convert(to)
        elif self.colorspace() is None:
            raise ValueError('Colorspace must be initialized by constructor or colorspace() to allow for colorspace conversion')
        else:
            raise ValueError('unsupported colorspace "%s"' % self.colorspace())

        self.colorspace(to)
        return self

    def affine_transform(self, A, border='zero'):
        """Apply a 3x3 affine geometric transformation to the image. 

        Args:        
            - A [np.ndarray]: 3x3 affine geometric transform from `vipy.geometry.affine_transform`
            - border [str]:  'zero' or 'replicate' to handle elements outside the image rectangle after transformation

        Returns:
            - This object with only the array transformed

        .. note:: The image will be loaded and converted to float() prior to applying the affine transformation.  
        .. note:: This will transform only the pixels
        """
        assert isnumpy(A) or isinstance(img, vipy.image.Image), "invalid input"
        assert A.shape == (3,3), "The affine transformation matrix should be the output of vipy.geometry.affine_transformation"
        self._array = vipy.geometry.imtransform(self.load().float().array(), A.astype(np.float32), border=border)
        return self

    def rotate(self, r):
        """Apply a rotation in radians to the pixels, with origin in upper left """
        return self.affine_transform(vipy.geometry.affine_transform(r=r))

    def rotate_by_exif(self):
        """Apply a rotation as specified in the 'Orientation' field EXIF metadata"""
        exif = self.exif()
        orientation = exif['Orientation'] if 'Orientation' in exif else None
        if orientation is None or orientation == 1:
            return self
        elif orientation == 2:
            return self.fliplr()
        elif orientation == 3:
            return self.flipud().fliplr()
        elif orientation == 4:
            return self.flipud()
        elif orientation == 5:
            return self.rot90cw().fliplr()
        elif orientation == 6:
            return self.rot90cw()
        elif orientation == 7:
            return self.rot90ccw().fliplr()
        elif orientation == 8:
            return self.rot90ccw()
        else:
            raise ValueError                        
    
    def rgb(self):
        """Convert the image buffer to three channel RGB uint8 colorspace"""
        return self._convert('rgb')

    def color_transform(self, colorspace):
        """Transform the image buffer from the current `vipy.image.Image.colorspace` to the provided colorspace"""
        return self._convert(colorspace)
    
    def colorspace_like(self, im):
        """Convert the image buffer to have the same colorspace as the provided image"""
        assert isinstance(im, vipy.image.Image)
        return self._convert(im.colorspace())
    
    def rgba(self):
        """Convert the image buffer to four channel RGBA uint8 colorspace"""
        return self._convert('rgba')

    def hsv(self):
        """Convert the image buffer to three channel HSV uint8 colorspace"""
        return self._convert('hsv')

    def bgr(self):
        """Convert the image buffer to three channel BGR uint8 colorspace"""
        return self._convert('bgr')

    def bgra(self):
        """Convert the image buffer to four channel BGR uint8 colorspace"""
        return self._convert('bgra')

    def float(self):
        """Convert the image buffer to float32"""
        return self._convert('float')

    def greyscale(self):
        """Convert the image buffer to single channel grayscale float32 in range [0,1]"""
        return self._convert('gray')

    def grayscale(self):
        """Alias for greyscale()"""
        return self.greyscale()

    def grey(self):
        """Alias for greyscale()"""
        return self.greyscale()

    def gray(self):
        """Alias for greyscale()"""
        return self.greyscale()

    def luminance(self):
        """Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component"""
        return self._convert('lum')

    def lum(self):
        """Alias for luminance()"""
        return self._convert('lum')

    def _apply_colormap(self, cm):
        """Convert an image to greyscale, then convert to RGB image with matplotlib colormap"""
        """https://matplotlib.org/tutorials/colors/colormaps.html"""
        cm = plt.get_cmap(cm)
        img = self.grey().numpy()
        self._array = np.uint8(255 * cm(img)[:,:,:3])
        self.colorspace('rgb')
        return self

    def jet(self):
        """Apply jet colormap to greyscale image and save as RGB"""
        return self._apply_colormap('jet')

    def rainbow(self):
        """Apply rainbow colormap to greyscale image and convert to RGB"""
        return self._apply_colormap('gist_rainbow')

    def hot(self):
        """Apply hot colormap to greyscale image and convert to RGB"""
        return self._apply_colormap('hot')

    def bone(self):
        """Apply bone colormap to greyscale image and convert to RGB"""
        return self._apply_colormap('bone')

    def saturate(self, min, max):
        """Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type"""
        return self.array(np.minimum(np.maximum(self.load().array(), min), max))

    def intensity(self):
        """Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'.  Equivalent to self.mat2gray()"""
        self.array((self.load().float().array()) - float(self.min()) / float(self.max() - self.min()))
        return self.colorspace('float')

    def mat2gray(self, min=None, max=None):
        """Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace.  This does not change the number of color channels"""
        self.array(mat2gray(np.float32(self.load().float().array()), min, max))
        return self.colorspace('float')
        return self

    def sum_to_one(self, eps=1E-6):
        """Return float image in the range [0,1] such that all elements sum to one"""
        return self.gain(1.0/(eps+self.mat2gray().sum()))
    
    def gain(self, g):
        """Elementwise multiply gain to image array, Gain should be broadcastable to array().  This forces the colospace to 'float'.  Don't use numba optimization, it is slower than native multiply"""
        return self.array(np.multiply(self.load().float().array(), g)).colorspace('float') if g != 1 else self

    def bias(self, b):
        """Add a bias to the image array.  Bias should be broadcastable to array().  This forces the colorspace to 'float'"""
        self.array(self.load().float().array() + b)
        return self.colorspace('float')

    def normalize(self, gain, bias):
        """Apply a multiplicative gain g and additive bias b, such that self.array() == gain*self.array() + bias.

        This is useful for applying a normalization of an image prior to calling `vipy.image.Image.torch`.

        The following operations are equivalent.

        ```python
        im = vipy.image.RandomImage()
        im.normalize(1/255.0, 0.5) == im.gain(1/255.0).bias(-0.5)
        ```
        
        .. note:: This will force the colorspace to 'float'
        """
        self.array(gain*self.load().float().array() + bias)
        return self.colorspace('float')

    def additive_noise(self, hue=(-15,15), saturation=(-15,15), brightness=(-15,15)):
        """Apply uniform random additive noise in the given range to the given HSV color channels.  Image will be converted to HSV prior to applying noise."""
        assert isinstance(hue, tuple) and len(hue) == 2 and hue[1]>=hue[0]
        assert isinstance(saturation, tuple) and len(saturation) == 2 and saturation[1]>=saturation[0]
        assert isinstance(brightness, tuple) and len(brightness) == 2 and brightness[1]>=brightness[0]        
        
        (H,W,C) = (self.height(), self.width(), self.channels())
        noise = np.dstack(((hue[1]-hue[0])*np.random.rand(H,W)+hue[0],
                           (saturation[1]-saturation[0])*np.random.rand(H,W)+saturation[0],
                           (brightness[1]-brightness[0])*np.random.rand(H,W)+brightness[0]))
        return self.array( np.minimum(np.maximum(self.hsv().array() + noise, 0), 255).astype(np.uint8) )
            
    # Image statistics
    def stats(self):
        log.info(self)
        log.info('  Channels: %d' % self.channels())
        log.info('  Shape: %s' % str(self.shape()))
        log.info('  min: %s' % str(self.min()))
        log.info('  max: %s' % str(self.max()))
        log.info('  mean: %s' % str(self.mean()))
        log.info('  channel mean: %s' % str(self.meanchannel()))        
    
    def min(self):
        return self.minpixel()

    def minpixel(self):
        return np.min(self.load().array().flatten())
    
    def max(self):
        return self.maxpixel()

    def maxpixel(self):
        return np.max(self.load().array().flatten())
    
    def mean(self):
        """Mean over all pixels"""
        return np.mean(self.load().array().flatten())

    def meanchannel(self, k=None):
        """Mean per channel over all pixels.  If channel k is provided, return just the mean for that channel"""
        C = np.mean(self.load().array(), axis=(0, 1)).flatten()
        return C[k] if k is not None else C
    
    def sum(self):
        return np.sum(self.load().array().flatten())

    # Image visualization
    def closeall(self):
        """Close all open figure windows"""
        vipy.show.closeall()
        return self
    
    def close(self, fignum=None):
        """Close the requested figure number, or close all of fignum=None"""
        if fignum is None:
            return self.closeall()
        else:
            vipy.show.close(fignum)
            return self
    
    def show(self, figure=1, nowindow=False, timestamp=None, timestampfacecolor='white', timestampcolor='black', mutator=None):
        """Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object"""
        assert self.load().isloaded(), 'Image not loaded'
        im = self.clone() if not mutator else mutator(self.clone())        
        vipy.show.imshow(im.rgb().numpy(), fignum=figure, nowindow=nowindow, timestamp=timestamp, timestampfacecolor=timestampfacecolor, flush=True, timestampcolor=timestampcolor)
        return self

    def save(self, filename, quality=75):
        """Save the current image to a new filename and return the image object"""
        assert filename is not None, "Invalid filename - must be path to new image filename"
        return self.filename(self.saveas(filename, quality=quality))
        
        
    # Image export
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

    def saveas(self, filename=None, writeas=None, quality=75):
        """Save current buffer (not including drawing overlays) to new filename and return filename.  If filename is not provided, use a temporary JPEG filename."""
        filename = tempjpg() if filename is None else filename
        if self.colorspace() in ['gray']:
            imwritegray(self.grayscale()._array, filename, quality=quality)
        elif self.colorspace() != 'float':
            imwrite(self.load().array(), filename, writeas=writeas, quality=quality)
        else:
            raise ValueError('Convert float image to RGB or gray first. Try self.mat2gray()')
        return filename

    def saveastmp(self):
        """Save current buffer to temp JPEG filename and return filename.  Alias for savetmp()"""
        return self.saveas(tempjpg())

    def savetmp(self):
        """Save current buffer to temp JPEG filename and return filename.   Alias for saveastmp()"""
        return self.saveastmp()

    def tocache(self):
        """Save current buffer to temp JPEG filename in the VIPY cache and return filename."""
        return self.saveas(vipy.util.tocache(tempjpg()))
    
    def base64(self):
        """Export a base64 encoding of the image suitable for embedding in an html page"""
        buf = io.BytesIO()
        self.clone().rgb().pil().save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue())
        
    def ascii(self):
        """Export a base64 ascii encoding of the image suitable for embedding in an <img> tag"""
        return self.base64().decode('ascii')

    def html(self, alt=None, id=None, attributes={'loading':'lazy'}):
        """Export a base64 encoding of the image suitable for embedding in an html page, enclosed in <img> tag
           
           Returns:
              -string:  <img src="data:image/jpeg;charset=utf-8;base64,%s" alt="%s" loading="lazy"> containing base64 encoded JPEG and alt text with lazy loading
        """
        assert isinstance(attributes, dict)
        b = self.base64().decode('ascii')
        alt_text = alt if alt is not None else self.filename()
        id = id if id is not None else self.filename()
        attr = ' '.join(['%s="%s"' % (str(k),str(v)) for (k,v) in attributes.items()])
        return '<img %ssrc="data:image/jpeg;charset=utf-8;base64,%s" alt="%s" %s>' % (('id="%s" ' % id) if id is not None else '', b, str(alt_text), attr)

    def annotate(self, timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None):
        """Change pixels of this image to include rendered annotation and return an image object"""
        # FIXME: for k in range(0,10): self.annotate().show(figure=k), this will result in cumulative figures
        return self.array(self.savefig(timestamp=timestamp, timestampcolor=timestampcolor, timestampfacecolor=timestampfacecolor, mutator=mutator, fontsize=fontsize).rgb().array()).downcast()

    def savefig(self, filename=None, figure=1, timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None):
        """Save last figure output from self.show() with drawing overlays to provided filename and return filename"""
        self.show(figure=figure, nowindow=True, timestamp=timestamp, timestampcolor=timestampcolor, timestampfacecolor=timestampfacecolor, mutator=mutator)  # sets figure dimensions, does not display window
        (W,H) = plt.figure(figure).canvas.get_width_height()  # fast
        buf = io.BytesIO()
        plt.figure(1).canvas.print_raw(buf)  # fast
        img = np.frombuffer(buf.getbuffer(), dtype=np.uint8).reshape((H, W, 4))  # RGBA
        vipy.show.close(figure)
        t = vipy.image.Image(array=img, colorspace='rgba')
        if filename is not None:
            t.rgb().saveas(os.path.abspath(os.path.expanduser(filename)))
        return t

    def map(self, func):
        """Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array().  The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'"""
        assert isinstance(func, types.LambdaType), "Input must be lambda function (e.g. f = lambda img: 255.0-img)"
        oldimg = self.array()  # reference
        newimg = func(self.array())  # in-place
        assert isnumpy(newimg), "Lambda function output must be numpy array"
        self.array(newimg)  # reference
        if newimg.dtype != oldimg.dtype or newimg.shape != oldimg.shape:
            self.colorspace('float')  # unknown colorspace after transformation, set generic
        return self

    def downcast(self):
        """Cast the class to the base class (vipy.image.Image)"""
        self.__class__ = vipy.image.Image
        return self

    def perceptualhash(self, bits=128, asbinary=False, asbytes=False):
        """Perceptual differential hash function

        This function converts to greyscale, resizes with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs.
        
        Args:
            bits: [int]  longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes).
            asbinary: [bool] If true, return a binary array
            asbytes: [bool] if true return a byte array

        Returns:
            A hash string encoding the perceptual hash such that `vipy.image.Image.perceptualhash_distance` can be used to compute a hash distance
            asbytes: a bytes array
            asbinary: a numpy binary array            

        .. notes::
            - Can be used for near duplicate detection by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing.  Equivalently, `vipy.image.Image.perceptualhash_distance`.
            - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex(h)
        """        
        allowablebits = [2*k*k for k in range(2, 17)]
        assert bits in allowablebits, "Bits must be in %s" % str(allowablebits)
        sq = int(np.ceil(np.sqrt(bits/2.0)))
        im = self.clone()
        b = (np.dstack(np.gradient(im.resize(cols=sq+1, rows=sq+1).greyscale().numpy()))[0:-1, 0:-1] > 0).flatten()
        return bytes(np.packbits(b)).hex() if not (asbytes or asbinary) else bytes(np.packbits(b)) if asbytes else b

    @staticmethod
    def perceptualhash_distance(h1, h2):
        """Hamming distance between two perceptual hashes"""
        assert len(h1) == len(h2)
        return np.sum(np.unpackbits(bytearray().fromhex(h1)) != np.unpackbits(bytearray().fromhex(h2)))
    

    def rot90cw(self):
        """Rotate the scene 90 degrees clockwise"""
        self.array(np.rot90(self.numpy(), 3))
        return self

    def rot90ccw(self):
        """Rotate the scene 90 degrees counterclockwise"""
        self.array(np.rot90(self.numpy(), 1))
        return self

    def face_detection(self, mindim=256, union=False, conf=0.2):
        """Detect faces in the scene, add as objects, return new scene with just faces
        
        Args:
            mindim [int]: The minimum dimension for downsampling the image for face detection.  Will be upsampled back to native resolution prior to return
            union [bool]: Whether to return a scene with just faces or the union of the existing scene and the new faces

        Returns
            A `vipy.image.Scene` object with all detected faces or the union of faces and all objects in self

        .. note:: This method uses a CPU-only pretrained face detector.  This is convenient, but slow.  See the heyvi package for optimized GPU batch processing for faster operation.
        """
        try_import('heyvi'); import heyvi  # >heyvi-0.2.28 for minconf      
        im = heyvi.detection.FaceDetector()(Scene.cast(self.clone()).clear().mindim(mindim)).mindim(self.mindim())
        return Scene.cast(self).union(im) if union else im
    
    def person_detection(self, mindim=256, union=False, conf=0.2):
        """Detect only people in the scene, add as objects, return new scene with just people

        Args:
            mindim [int]: The minimum dimension for downsampling the image for person detection.  Will be upsampled back to native resolution prior to return
            union [bool]: Whether to return a scene with just people or the union of the existing scene and the new people
            conf [float]: A real value between [0,1] of the minimum confidence for person detection

        Returns
            A `vipy.image.Scene` object with all detected people or the union of people and all objects in self
        
        .. note:: This method uses a CPU-only pretrained person detector.  This is convenient, but slow.  See the heyvi package for optimized GPU batch processing for faster operation.
        """
        try_import('heyvi'); import heyvi                
        im = heyvi.detection.ObjectDetector()(Scene.cast(self.clone()).clear().mindim(mindim), conf=conf, objects=['person']).mindim(self.mindim())
        return Scene.cast(self).union(im) if union else im        

    def qrcode_recognition(self):
        """Detect and decode one QR code in the current image using OpenCV, and return the string contents of the QR code or None if the detection failed to detect a code.  (Requires OpenCV)"""
        try_import('cv2'); import cv2
        (value, corners, rectified) = cv2.QRCodeDetector().detectAndDecode(self.load().array())
        return value if len(value)>0 else None
    
    def faceblur(self, radius=4, mindim=256):
        """Replace pixels for all detected faces with `vipy.image.Scene.blurmask`, add locations of detected faces into attributes.

        Args:
            radius [int]: The radius of pixels for `vipy.image.Scene.blurmask`
            mindim [int]: The minimum dimension for downsampling the image for face detection.  Will be upsampled prior to pixelize.
        
        Returns:
            A `vipy.image.Image` object with a pixel buffer with all faces pixelized, with faceblur attribute set in `vipy.image.Image.metadata` showing the locations of the blurred faces.

        .. notes::
            - This method uses a CPU-only pretrained torch network for face detection from the heyvi visual analytics package, which is re-initialized on each call to this method.  
            - For batch operations on many images, it is preferred to set up the detection network once, then calling many images sequentially.  
            - To retain boxes, use self.face_detection().blurmask()
        """
        im = self.face_detection(mindim=mindim, union=False)  # only faces
        return im.setattribute('faceblur', [o.int().json() for o in im.objects()]).blurmask(radius=radius).downcast()

    def facepixelize(self, radius=7, mindim=256):
        """Replace pixels for all detected faces with `vipy.image.Scene.pixelize`, add locations of detected faces into attributes.

        Args:
            radius [int]: The radius of pixels for `vipy.image.Scene.radius`
            mindim [int]: The minimum dimension for downsampling the image for face detection.  Will be upsampled prior to pixelize.
        
        Returns:
            A `vipy.image.Image` object with a pixel buffer with all faces pixelized, with facepixelize attribute set in `vipy.image.Image.metadata` showing the locations of the blurred faces.

        .. notes::
            - This method uses a CPU-only pretrained torch network for face detection from the heyvi visual analytics package, which is re-initialized on each call to this method.  
            - For batch operations on many images, it is preferred to set up the detection network once, then calling many images sequentially.  
            - To retain boxes, use self.face_detection().pixelize()
        """
        im = self.face_detection(mindim=mindim, union=False)          
        return im.setattribute('facepixelize', [o.int().json() for o in im.objects()]).pixelize(radius=radius).downcast()
    
    
class ImageCategory(Image):
    """vipy ImageCategory class

    This class provides a representation of a vipy.image.Image with a category. 

    Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label')

    ```python
    im = vipy.image.ImageCategory(filename='/path/to/dog_image.ext', category='dog')
    im = vipy.image.ImageCategory(url='http://path/to/dog_image.ext', category='dog')
    im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog')
    ```

    """
    
    def __init__(self, filename=None, url=None, category=None, label=None, attributes=None, array=None, colorspace=None):
        # Image class inheritance
        super().__init__(filename=filename,
                         url=url,
                         attributes=attributes,
                         array=array,
                         colorspace=colorspace)
        assert not (category is not None and label is not None), "Define either category or label kwarg, not both"
        self._category = category if category is not None else label

    @classmethod
    def cast(cls, im, flush=False):
        assert isinstance(im, vipy.image.Image)
        im.__class__ = vipy.image.ImageCategory
        im._category = None if flush or not hasattr(im, '_category') else im._category
        return im

    @classmethod
    def from_json(obj, s):
        im = super().from_json(s)
        d = {k.lstrip('_'):v for (k,v) in (json.loads(s) if not isinstance(s, dict) else s).items()}  # prettyjson (remove "_" prefix to attributes)                    
        im._category = d['category']
        return im

    def json(self, s=None, encode=True):
        if s is None:
            d = json.loads(super().json())
            d['category'] = self._category if not isinstance(self._category, set) else list(self._category)  # coerce set -> list
            return json.dumps(d) if encode else d
        else:
            super().json(s)
            d = json.loads(s)            
            self._category = d['category'] 
            return self
    
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        if self.filename() is not None:
            strlist.append('filename="%s"' % (self.filename()))
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if not self.isloaded() and self.has_loader() is not None:
            strlist.append('loaded=False')            
        if self.category() is not None and len(str(self.category()))>0:
            strlist.append('category=%s' % (str(self.category())[0:80] + (' ... ' if len(str(self.category()))>80 else '')))
        return str('<vipy.image.ImageCategory: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        return self._category == other._category if isinstance(other, ImageCategory) else False

    def __ne__(self, other):
        return self._category != other._category

    def is_(self, other):
        return self.__eq__(other)

    def is_not(self, other):
        return self.__ne__(other)

    def nocategory(self):
        self._category = None
        return self

    def category(self, newcategory=None):
        """Return or update the category"""
        if newcategory is None:
            return self._category
        else:
            self._category = newcategory
            return self

    def label(self, newlabel=None):
        """Alias for category"""
        return self.category(newlabel)
    
    def score(self, newscore=None):
        """Real valued score for categorization, larger is better"""
        if newscore is None:
            return self.getattribute('score')
        else:
            self.setattribute('score', newscore)
            return self

    def probability(self, newprob=None):
        """Real valued probability for categorization, [0,1]"""
        if newprob is None:
            return self.getattribute('probability')
        else:
            self.setattribute('probability', newprob)
            self.setattribute('RawDetectionProbability', newprob)
            return self
    

class LabeledImage(Image):
    """vipy.image.LabeledImage class

    This class provides a representation of a vipy.image.Image with one or more labels.

    Valid constructors include all provided by vipy.image.Image with additional labels that provide ground truth for the content of the image. 

    Suppprted labels are tags/keywords, categories, captions, image identifier, wordnet ID, visual Q&A 

    ```python
    im = vipy.image.LabeledImage(filename='/path/to/dog.jpg', tags={'dog','canine'})
    im = vipy.image.LabeledImage(filename='/path/to/dog.jpg', caption=['this is a dog.', 'this image contains a canine']))
    im = vipy.image.LabeledImage(url='https://here.com/dog.jpg', instanceid=0, category='dog', vqa=[{'question':'Is this a dog?', 'answer':'Yes'}])
    ```
    """
    def __init__(self, filename=None, url=None, attributes=None, array=None, colorspace=None, tags=None, caption=None, instanceid=None, category=None, wordnetid=None, vqa=None):
        super().__init__(filename=filename,
                         url=url,
                         attributes=attributes,
                         array=array,
                         colorspace=colorspace)
        
        if not self.hasattribute('label'):
            self.attributes['label'] = {}
        if tags is not None:
            self.attributes['label']['tags'] = sorted(list(set(vipy.util.tolist(tags))))            
        if caption is not None:
            self.attributes['label']['caption'] = vipy.util.tolist(caption)
        if instanceid is not None:
            self.attributes['label']['instanceid'] = instanceid
        if category is not None:
            self.attributes['label']['category'] = category
        if vqa is not None:
            assert isinstance(vqa, list) and all([isinstance(q, dict) and 'question' in q and 'answer' in q for q in vqa])
            self.attributes['label']['vqa'] = vqa
            
    def __repr__(self):
        fields = ["height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace())] if self.isloaded() else []
        fields += ['filename="%s"' % (self.filename())] if self.filename() is not None else []
        fields += ['url="%s"' % self.url()] if self.hasurl() else []
        fields += ['loaded=False'] if not self.isloaded() and self.has_loader() is not None else []
        fields += ['category=%s' % vipy.util.truncate_string(self.category(), 40)] if self.has_category() else []
        fields += ['iid=%s' % self.instanceid()] if self.has_instanceid() else []
        fields += ['tags=%s' % vipy.util.truncate_string(str(self.tags()), 40)] if self.has_tags() else []
        fields += ['caption=%s' % vipy.util.truncate_string(str(self.caption()), 40)] if self.has_caption() else []
        fields += ['vqa=%s' % vipy.util.truncate_string(str(self.vqa()), 40)] if self.has_vqa() else []
        return str('<vipy.image.LabeledImage: %s>' % (', '.join(fields)))
        
    @classmethod
    def cast(cls, im, tags=None, caption=None, instanceid=None, category=None, wordnetid=None, vqa=None):
        assert isinstance(im, vipy.image.Image)

        if not im.hasattribute('label'):
            im.attributes['label'] = {}
        if isinstance(im, vipy.image.ImageCategory):
            im.attributes['label']['category'] = im.category()            
        if tags is not None:
            im.attributes['label']['tags'] = sorted(list(set(vipy.util.tolist(tags))))
        if caption is not None:
            im.attributes['label']['caption'] = vipy.util.tolist(caption)
        if instanceid is not None:
            im.attributes['label']['instanceid'] = instanceid
        if category is not None:
            im.attributes['label']['category'] = category
        if vqa is not None:
            assert isinstance(vqa, list) and all([isinstance(q, dict) and 'question' in q and 'answer' in q for q in vqa])
            im.attributes['label']['vqa'] = vqa        
        im.__class__ = vipy.image.LabeledImage
        
        return im

    def tags(self):
        return set(self.attributes['label']['tags'])
    def has_tag(self, t):
        return t in self.attributes['label']['tags']
    def has_tags(self):
        return 'tags' in self.attributes['label']
    
    def has_caption(self):
        return 'caption' in self.attributes['label']
    def caption(self):
        return self.attributes['label']['caption'] if self.has_caption() else None
    
    def has_instanceid(self):
        return 'instanceid' in self.attributes['label']
    def instanceid(self, iid=None):
        if iid is not None:
             self.attributes['label']['instanceid'] = iid
             return self
        return self.attributes['label']['instanceid'] if self.has_instanceid() else None

    def has_wordnetid(self):
        return 'wordnetid' in self.attributes['label']
    def wordnetid(self):
        return self.attributes['label']['wordnetid'] if self.has_wordnetid() else None

    def has_category(self):
        return 'category' in self.attributes['label'] and self.attributes['label']['category'] is not None    
    def category(self, new=None):
        if new is not None:
            self.attributes['label']['category'] = new
            return self
        return self.attributes['label']['category'] if self.has_category() else None
    
    def del_category(self):
        self.attributes['label']['category'] = None
        return self
    
    
    def has_vqa(self):
        return 'vqa' in self.attributes['label']     
    def vqa(self):
        return self.attributes['label']['vqa'] if self.has_vqa() else None   

    def is_unlabeled(self):
        return len(self.attributes['label']) == 0
    
    
class Scene(ImageCategory):
    """vipy.image.Scene class

    This class provides a representation of a vipy.image.ImageCategory with one or more vipy.object.Object.  The goal of this class is to provide a unified representation for all objects in a scene.

    Valid constructors include all provided by vipy.image.Image() and vipy.image.ImageCategory() with the additional kwarg 'objects', which is a list of vipy.object.Object()

    ```python
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='city', objects=[vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='city').objects([vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels='face', xywh=[0,0,100,100])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels='face', xywh=[[0,0,100,100], [100,100,200,200]])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels=['face', 'desk'] xywh=[[0,0,100,100], [200,200,300,300]])
    ```

    """
    def __init__(self, filename=None, url=None, category=None, attributes=None, objects=None, xywh=None, boxlabels=None, array=None, colorspace=None):
        super().__init__(filename=filename, url=url, attributes=attributes, category=category, array=array, colorspace=colorspace)   # ImageCategory class inheritance
        self._objectlist = []

        if objects is not None:
            if not (isinstance(objects, list) and all([isinstance(bb, vipy.object.Object) for bb in objects])):
                raise ValueError("Invalid object list - Input must be [vipy.object.Object, ...]")
            self._objectlist = objects

        detlist = []
        if xywh is not None:
            if (islistoflists(xywh) or istupleoftuples(xywh)) and all([len(bb)==4 for bb in xywh]):
                detlist = [vipy.object.Detection(category=None, xywh=bb) for bb in xywh]
            elif (islist(xywh) or isinstance(xywh, tuple)) and len(xywh)==4 and all([isnumber(bb) for bb in xywh]):
                detlist = [vipy.object.Detection(category=None, xywh=xywh)]
            else:
                raise ValueError("Invalid xywh list - Input must be [[x1,y1,w1,h1], ...")            
        if boxlabels is not None:
            if isstring(boxlabels):
                label = boxlabels
                detlist = [d.category(label) for d in detlist]
            elif (isinstance(boxlabels, tuple) or islist(boxlabels)) and len(boxlabels) == len(xywh):
                detlist = [d.category(label) for (d,label) in zip(detlist, boxlabels)]
            else:
                raise ValueError("Invalid boxlabels list - len(boxlabels) must be len(xywh) with corresponding labels for each xywh box  [label1, label2, ...]")

        self._objectlist = self._objectlist + detlist

    @classmethod
    def cast(cls, im):
        assert isinstance(im, vipy.image.Image), "Invalid input - must be derived from vipy.image.Image"
        if im.__class__ != vipy.image.Scene:
            im.__class__ = vipy.image.Scene
            im._category = None if not hasattr(im, '_category') else im._category
            im._objectlist = [] if not hasattr(im, '_objectlist') else im._objectlist  
        return im
    
    @classmethod
    def from_json(obj, s):
        im = super().from_json(s)
        d = {k.lstrip('_'):v for (k,v) in (json.loads(s) if not isinstance(s, dict) else s).items()}  # prettyjson (remove "_" prefix to attributes)
        if isinstance(d['objectlist'], dict):
            # Version 1.15.1: expanded serialization to support multiple object types
            im._objectlist = [vipy.object.Detection.from_json(s) for s in d['objectlist']['Detection']] + [vipy.object.Point2d.from_json(s) for s in d['objectlist']['Point2d']]
        else:
            # Legacy support: 
            im._objectlist = [vipy.object.Detection.from_json(s) for s in d['objectlist']]            
        return im

    def __json__(self):
        """Serialization method for json package"""
        return self.json(encode=True)
    
    def json(self, s=None, encode=True):
        if s is None:
            d = json.loads(super().json())
            d['objectlist'] = {'Detection': [bb.json(encode=False) for bb in self._objectlist if isinstance(bb, vipy.object.Detection)],
                               'Point2d': [p.dict() for p in self._objectlist if isinstance(p, vipy.object.Point2d)]}            
            return json.dumps(d) if encode else d
        else:
            super().json(s)
            d = json.loads(s)            
            self._objectlist = [vipy.object.Detection.from_json(s) for s in d['objectlist']['Detection']] + [vipy.object.Point2d.from_json(s) for s in d['objectlist']['Point2d']]
            return self
        
    def __eq__(self, other):
        """Scene equality requires equality of all objects in the scene, assumes a total order of objects"""
        return isinstance(other, Scene) and len(self)==len(other) and all([obj1 == obj2 for (obj1, obj2) in zip(self, other)])

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        if self.filename() is not None:
            strlist.append('filename="%s"' % (self.filename()))
        if self.hasurl():
            strlist.append('url=%s' % self.url())
        if self.category() is not None:
            strlist.append('category=%s' % (str(self.category())[0:80] + (' ... ' if len(str(self.category()))>80 else '')))
        if len(self.objects()) > 0:
            strlist.append('objects=%d' % len(self.objects()))
        return str('<vipy.image.scene: %s>' % (', '.join(strlist)))

    def __len__(self):
        """The length of a scene is equal to the number of objects present in the scene"""
        return len(self._objectlist)

    def __iter__(self):
        """Iterate over each ImageDetection() in the scene"""
        for (k, im) in enumerate(self._objectlist):
            yield self.__getitem__(k)

    def __getitem__(self, k):
        """Return the kth object in the scene as a `vipy.image.ImageDetection` object """
        assert isinstance(k, int), "Indexing by object in scene must be integer"
        obj = self._objectlist[k].clone()
        return ImageDetection(array=self.array(), filename=self.filename(), url=self.url(), colorspace=self.colorspace(), xmin=obj.xmin(), ymin=obj.ymin(), width=obj.width(), height=obj.height(), category=obj.category(), attributes=obj.attributes, id=obj.id())

    def split(self):
        """Split a scene with K objects into a list of K `vipy.image.ImageDetection` objects, each with one object in the scene.
        
        .. note:: The pixel buffer is shared between each split.  Use [im.clone() for im in self.split()] for an explicit copy.
        """
        return list(self)
    
    def append(self, imdet):
        """Append the provided vipy.object.Detection object to the scene object list"""
        assert isinstance(imdet, vipy.object.Object), "Invalid input"
        self._objectlist.append(imdet)
        return self

    def add(self, imdet):
        """Alias for append"""        
        return self.append(imdet)
    
    def objects(self, objectlist=None):
        if objectlist is None:
            return self._objectlist
        else:
            assert isinstance(objectlist, list) and (len(objectlist) == 0 or all([isinstance(bb, vipy.object.Object) for bb in objectlist])), "Invalid object list"
            self._objectlist = objectlist
            return self

    def objectmap(self, f):
        """Apply lambda function f to each object.  If f is a list of lambda, apply one to one with the objects"""
        assert callable(f)
        self._objectlist = [f(obj)  for obj in self._objectlist] if not isinstance(f, list) else [g(obj) for (g,obj) in zip(f, self._objectlist)]
        assert all([isinstance(a, vipy.object.Object) for a in self.objects()]), "Lambda function must return vipy.object.Detection"
        return self

    def objectfilter(self, f):
        """Apply lambda function f to each object and keep if filter is True"""
        assert callable(f)
        self._objectlist = [obj for obj in self._objectlist if f(obj) is True]
        return self

    def nms(self, conf, iou, cover=0.8):
        """Non-maximum supporession of objects() by category based on confidence and spatial IoU and cover thresholds"""
        return self.objects( vipy.object.non_maximum_suppression(self.objects(), conf=conf, iou=iou, cover=cover, bycategory=True) )

    def intersection(self, other, miniou, bycategory=True):
        """Return a Scene() containing the objects in both self and other, that overlap by miniou with greedy assignment"""
        assert isinstance(other, Scene), "Invalid input"
        v = self.clone()
        v._objectlist = [v._objectlist[k] for (k,d) in enumerate(greedy_assignment(v.objects(), other.objects(), miniou, bycategory=bycategory)) if d is not None]
        return v

    def difference(self, other, miniou):
        """Return a Scene() containing the objects in self but not other, that overlap by miniou with greedy assignment"""
        assert isinstance(other, Scene), "Invalid input"
        v = self.clone()
        v._objectlist = [v._objectlist[k] for (k,d) in enumerate(greedy_assignment(self.objects(), other.objects(), miniou, bycategory=True)) if d is None]
        return v
        
    def union(self, other, miniou=None):
        """Combine the objects of the scene with other and self with no duplicate checking unless miniou is not None"""
        if isinstance(other, Scene):
            self.objects(self.objects()+other.objects())
        return self

    def uncrop(self, bb, shape):
        """Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)"""
        super().uncrop(bb, shape)
        return self.objectmap(lambda o: o.translate(bb.xmin(), bb.ymin()))
        
    def clear(self):
        """Remove all objects from this scene."""
        return self.objects([])
    
    def boundingbox(self):
        """The boundingbox of a scene is the union of all object bounding boxes, or None if there are no objects"""
        boxes = [bb for bb in self.objects() if isinstance(bb, vipy.geometry.BoundingBox)]
        bb = boxes[0].clone() if len(boxes) >= 1 else None
        return bb.union(boxes[1:]) if len(boxes) >= 2 else bb

    def categories(self):
        """Return list of unique object categories in scene"""
        return list(set([obj.category() for obj in self._objectlist]))
    
    # Spatial transformation
    def imclip(self):
        """Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image"""
        self._objectlist = [o.imclip(self.numpy()) if isinstance(o, vipy.geometry.BoundingBox) else o for o in self._objectlist if not isinstance(o, vipy.geometry.BoundingBox) or o.hasoverlap(self.numpy())]
        return self

    def rescale(self, scale=1, interp='bilinear'):
        """Rescale image buffer and all bounding boxes - Not idempotent"""
        self = super().rescale(scale, interp=interp)
        self._objectlist = [bb.rescale(scale) for bb in self._objectlist]
        return self

    def resize(self, cols=None, rows=None, height=None, width=None, interp='bilinear'):
        """Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly.  If cols or rows is None, then scale isotropically"""
        assert not (cols is not None and width is not None), "Define either width or cols"
        assert not (rows is not None and height is not None), "Define either height or rows"
        rows = rows if height is None else height
        cols = cols if width is None else width        
        assert cols is not None or rows is not None, "Invalid input"
        
        sx = (float(cols) / self.width()) if cols is not None else None
        sy = (float(rows) / self.height()) if rows is not None else None
        sx = sy if sx is None else sx
        sy = sx if sy is None else sy        
        self._objectlist = [bb.scalex(sx).scaley(sy) for bb in self._objectlist]        
        if sx == sy:
            self = super().rescale(sx, interp=interp)  # FIXME: if we call resize here, inheritance is screweed up
        else:
            self = super().resize(cols, rows, interp=interp)
        return self

    def centersquare(self):
        """Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W)) preserving center, and update bounding boxes"""
        (H,W) = self.shape()
        self = super().centersquare()
        (dy, dx) = ((H - self.height())/2.0, (W - self.width())/2.0)
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]
        return self
    
    def fliplr(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self._objectlist = [bb.fliplr(self.numpy()) for bb in self._objectlist]
        self = super().fliplr()
        return self

    def flipud(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self._objectlist = [bb.flipud(self.numpy()) for bb in self._objectlist]
        self = super().flipud()
        return self
    
    def dilate(self, s):
        """Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle"""
        self._objectlist = [bb.dilate(s) for bb in self._objectlist]
        return self

    def zeropad(self, padwidth, padheight):
        """Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super().zeropad(padwidth, padheight)
        dx = padwidth[0] if isinstance(padwidth, tuple) and len(padwidth) == 2 else padwidth
        dy = padheight[0] if isinstance(padheight, tuple) and len(padheight) == 2 else padheight
        self._objectlist = [bb.translate(dx, dy) for bb in self._objectlist]
        return self

    def meanpad(self, padwidth, padheight, mu=None):
        """Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super().meanpad(padwidth, padheight, mu=mu)
        dx = padwidth[0] if isinstance(padwidth, tuple) and len(padwidth) == 2 else padwidth
        dy = padheight[0] if isinstance(padheight, tuple) and len(padheight) == 2 else padheight
        self._objectlist = [bb.translate(dx, dy) for bb in self._objectlist]
        return self

    def rot90cw(self):
        """Rotate the scene 90 degrees clockwise, and update objects"""
        (H,W) = self.shape()        
        self.array(np.rot90(self.numpy(), 3))
        self._objectlist = [bb.rot90cw(H, W) for bb in self._objectlist]
        return self

    def rot90ccw(self):
        """Rotate the scene 90 degrees counterclockwise, and update objects"""
        (H,W) = self.shape()
        self.array(np.rot90(self.numpy(), 1))
        self._objectlist = [bb.rot90ccw(H, W) for bb in self._objectlist]
        return self

    def maxdim(self, dim=None, interp='bilinear'):
        """Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects"""
        return super().maxdim(dim, interp=interp) if dim is not None else max(self.shape())  # will call self.rescale() which will update boxes

    def mindim(self, dim=None, interp='bilinear'):
        """Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects"""
        return super().mindim(dim, interp=interp) if dim is not None else min(self.shape())  # will call self.rescale() which will update boxes

    def crop(self, bbox=None):
        """Crop the image buffer using the supplied bounding box object (or the only object if bbox=None), clipping the box to the image rectangle, update all scene objects"""
        assert bbox is not None or (len(self) == 1), "Bounding box must be provided if number of objects != 1"
        bbox = bbox if bbox is not None else [o for o in self._objectlist if isinstance(o, vipy.geometry.BoundingBox)][0]
        self = super()._crop(bbox)        
        (dx, dy) = (bbox.xmin(), bbox.ymin())
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]
        return self

    def objectcrop(self, dilate=1.0, maxsquare=False):
        """Crop image using the `vipy.image.Scene.boundingbox` with dilation factor, setting to maxsquare prior to crop as requested.  Crop will be zeropadded if outside the image rectangle."""
        bb = self.boundingbox()
        return self.padcrop(bb.dilate(dilate).maxsquareif(maxsquare)) if bb is not None else self
    
    def centercrop(self, height, width):
        """Crop image of size (height x width) in the center, keeping the image centroid constant"""
        return self.crop(BoundingBox(xcentroid=float(self.width() / 2.0), ycentroid=float(self.height() / 2.0), width=int(width), height=int(height)))

    def cornercrop(self, height, width):
        """Crop image of size (height x width) from the upper left corner, returning valid pixels only"""
        return self.crop(BoundingBox(xmin=0, ymin=0, width=int(width), height=int(height)))
    
    def padcrop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects"""
        self.zeropad(bbox.int().width(), bbox.int().height())  # FIXME: this is inefficient
        (dx, dy) = (bbox.width(), bbox.height())
        bbox = bbox.translate(dx, dy)
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]        
        self = super()._crop(bbox)        
        (dx, dy) = (bbox.xmin(), bbox.ymin())
        return self

    def cornerpadcrop(self, height, width):
        """Crop image of size (height x width) from the upper left corner, returning zero padded result out to (height, width)"""
        return self.padcrop(BoundingBox(xmin=0, ymin=0, width=width, height=height))
    
    # Image export
    def rectangular_mask(self, W=None, H=None):
        """Return a binary array of the same size as the image (or using the
        provided image width and height (W,H) size to avoid an image load),
        with ones inside all bounding boxes"""
        if (W is None or H is None):
            (H, W) = (int(np.round(self.height())),
                      int(np.round(self.width())))
        immask = np.zeros((H, W)).astype(np.uint8)
        for bb in self._objectlist:
            if isinstance(bb, vipy.geometry.BoundingBox) and bb.hasoverlap(immask):
                bbm = bb.clone().imclip(self.numpy()).int()
                immask[bbm.ymin():bbm.ymax(), bbm.xmin():bbm.xmax()] = 1
        return immask

    def binarymask(self):
        """Alias for rectangular_mask with in-place update"""
        mask = self.rectangular_mask() if self.channels() == 1 else np.expand_dims(self.rectangular_mask(), axis=2)
        img = self.numpy()
        img[:] = mask[:]  # in-place update
        return self
        
    def bgmask(self):
        """Set all pixels outside the bounding box to zero"""
        mask = self.rectangular_mask() if self.channels() == 1 else np.expand_dims(self.rectangular_mask(), axis=2)
        img = self.numpy()
        img[:] = np.multiply(img, mask)  # in-place update
        return self  

    def fgmask(self):
        """Set all pixels inside the bounding box to zero"""
        mask = self.rectangular_mask() if self.channels() == 1 else np.expand_dims(self.rectangular_mask(), axis=2)
        img = self.numpy()
        img[:] = np.multiply(img, 1.0-mask)  # in-place update
        return self
    
    def pixelmask(self, pixelsize=8):
        """Replace pixels within all foreground objects with a privacy preserving pixelated foreground with larger pixels (e.g. like privacy glass)"""
        assert pixelsize > 1, "Pixelsize is a scale factor such that pixels within the foreground are pixelsize times larger than the background"
        (img, mask) = (self.numpy(), self.rectangular_mask())  # force writeable
        img[mask > 0] = self.clone().rescale(1.0/pixelsize, interp='nearest').resize_like(self, interp='nearest').numpy()[mask > 0]  # in-place update
        return self

    def pixelize(self, radius=16):
        """Alias for pixelmask"""
        return self.pixelmask(pixelsize=radius)
    def pixelate(self, radius=16):
        """Alias for pixelmask"""
        return self.pixelmask(pixelsize=radius)
        
    
    def blurmask(self, radius=7):
        """Replace pixels within all foreground objects with a privacy preserving blurred foreground"""
        assert radius > 1, "Pixelsize is a scale factor such that pixels within the foreground are pixelsize times larger than the background"
        (img, mask) = (self.numpy(), self.rectangular_mask())  # force writeable
        img[mask > 0] = self.clone().blur(radius).numpy()[mask > 0]  # in-place update
        return self

    def blurmask_only(self, categories, radius=7):
        """Replace pixels within all foreground objects with specified category with a privacy preserving blurred foreground"""
        assert radius > 1, "Pixelsize is a scale factor such that pixels within the foreground are pixelsize times larger than the background"

        objects = self.objects()
        return self.clone().objects([o for o in objects if o.category() in categories]).blurmask(radius=radius).objects(objects)
    
    def replace(self, newim, broadcast=False):
        """Set all image values within the bounding box equal to the provided img, triggers load() and imclip()"""
        assert isinstance(newim, vipy.image.Image), "Invalid replacement image - Must be vipy.image.Image"
        img = self.numpy()        
        newimg = newim.array()
        for d in self._objectlist:
            d.imclip(newimg).imclip(img)
            img[int(d.ymin()):int(d.ymax()),
                int(d.xmin()):int(d.xmax())] = newimg[int(d.ymin()):int(d.ymax()),
                                                      int(d.xmin()):int(d.xmax())] if not broadcast else newim.clone().resize(int(d.width()), int(d.height())).array()
        return self
    
    def meanmask(self):
        """Replace pixels within the foreground objects with the mean pixel color"""
        img = self.numpy()  # force writeable
        img[self.rectangular_mask() > 0] = self.meanchannel()  # in-place update
        return self

    
    def perceptualhash(self, bits=128, asbinary=False, asbytes=False, objmask=False):
        """Perceptual differential hash function.

        This function sets foreground objects to mean color, convert to greyscale, resize with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs.
        
        Args:
            bits: [int]  longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes).
            objmask: [bool] if true, replace the foreground object masks with the mean color prior to computing
            asbinary: [bool] If true, return a binary array
            asbytes: [bool] if true return a byte array

        Returns:
            A hash string encoding the perceptual hash such that `vipy.image.Image.perceptualhash_distance` can be used to compute a hash distance
            asbytes: a bytes array
            asbinary: a numpy binary array            

        .. notes::
            - Can be used for near duplicate detection of background scenes by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing.  Equivalently, `vipy.image.Image.perceptualhash_distance`.
            - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex( bghash() )) which is equivalent to perceptualhash(asbinary=True)
       
        """        
        allowablebits = [2*k*k for k in range(2, 17)]
        assert bits in allowablebits, "Bits must be in %s" % str(allowablebits)
        sq = int(np.ceil(np.sqrt(bits/2.0)))
        im = self.clone() if not objmask else self.clone().meanmask()        
        b = (np.dstack(np.gradient(im.resize(cols=sq+1, rows=sq+1).greyscale().numpy()))[0:-1, 0:-1] > 0).flatten()
        return bytes(np.packbits(b)).hex() if not (asbytes or asbinary) else bytes(np.packbits(b)) if asbytes else b

    def fghash(self, bits=8, asbinary=False, asbytes=False):
        """Perceptual differential hash function, computed for each foreground region independently"""
        return [im.crop().perceptualhash(bits=bits, asbinary=asbinary, asbytes=asbytes, objmask=False)  for im in self]

    
    def bghash(self, bits=128, asbinary=False, asbytes=False):
        """Percetual differential hash function, masking out foreground regions"""
        return self.clone().greyscale().perceptualhash(bits=bits, asbinary=asbinary, asbytes=asbytes, objmask=True)
        
    def isduplicate(self, im, threshold, bits=128):
        """Background hash near duplicate detection, returns true if self and im are near duplicate images using bghash"""
        assert isinstance(im, Image), "Invalid input"
        return vipy.image.Image.perceptualhash_distance(self.bghash(bits=bits), im.bghash(bits=bits)) < threshold 
    
        
    def show(self, categories=None, figure=1, nocaption=False, nocaption_withstring=[], fontsize=10, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, captionoffset=(0,0), nowindow=False, textfacecolor='white', textfacealpha=1.0, shortlabel=True, timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None, timestampoffset=(0,0)):
        """Show scene detection 

        Args:
           - categories: [list]  List of category (or shortlabel) names in the scene to show
           - fontsize: [int] or [str]: Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size
           - figure: [int] Figure number, show the image in the provided figure=int numbered window
           - nocaption: [bool]  Show or do not show the text caption in the upper left of the box 
           - nocaption_withstring: [list]:  Do not show captions for those object categories (or shortlabels) containing any of the strings in the provided list
           - boxalpha (float, [0,1]):  Set the text box background to be semi-transparent with an alpha
           - d_category2color (dict):  Define a dictionary of required mapping of specific category() to box colors.  Non-specified categories are assigned a random named color from vipy.show.colorlist()
           - caption_offset (int, int): The relative position of the caption to the upper right corner of the box.
           - nowindow (bool):  Display or not display the image
           - textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background
           - textfacealpha (float, [0,1]):  The textbox background transparency
           - shortlabel (bool):  Whether to show the shortlabel or the full category name in the caption
           - mutator (lambda):  A lambda function with signature lambda im: f(im) which will modify this image prior to show.  Useful for changing labels on the fly
           - timestampoffset (tuple): (x,y) coordinate offsets to shift the upper left corner timestamp
        """
        colors = vipy.show.colorlist()
        im = self.clone() if not mutator else mutator(self.clone())

        valid_objects = [obj.clone() for obj in im._objectlist if categories is None or obj.category() in tolist(categories)]  # Objects with valid category
        valid_objects = [obj.imclip(self.numpy()) for obj in valid_objects if obj.hasoverlap(self.numpy())]  # Objects within image rectangle
        valid_objects = [obj.category(obj.shortlabel()) if obj.shortlabel() is not None else obj for obj in valid_objects] if shortlabel else valid_objects  # Display name as shortlabel?
        d_categories2color = {d.category():colors[int(hashlib.sha1(d.category().split(' ')[-1].encode('utf-8')).hexdigest(), 16) % len(colors)] for d in valid_objects}   # consistent color mapping by category suffix (space separated)
        d_categories2color.update(d_category2color)  # requested color mapping
        object_color = [d_categories2color[d.category()] for d in valid_objects]                
        valid_objects  = [d if not any([c in d.category() for c in tolist(nocaption_withstring)]) else d.nocategory() for d in valid_objects]  # Objects requested to show without caption
        imdisplay = self.clone().rgb() if self.colorspace() != 'rgb' else self.load()  # convert to RGB for show() if necessary
        fontsize_scaled = float(fontsize.split(':')[0])*(min(imdisplay.shape())/640.0) if isstring(fontsize) else fontsize
        imdisplay = mutator(imdisplay) if mutator is not None else imdisplay
        vipy.show.imobjects(imdisplay._array, valid_objects, bordercolor=object_color, textcolor=object_color, fignum=figure, do_caption=(nocaption==False), facealpha=boxalpha, fontsize=fontsize_scaled,
                            captionoffset=captionoffset, nowindow=nowindow, textfacecolor=textfacecolor, textfacealpha=textfacealpha, timestamp=timestamp, timestampcolor=timestampcolor, timestampfacecolor=timestampfacecolor, timestampoffset=timestampoffset)
        return self

    def annotate(self, outfile=None, categories=None, figure=1, nocaption=False, fontsize=10, boxalpha=0.25, d_category2color={'person':'green', 'vehicle':'blue', 'object':'red'}, captionoffset=(0,0), dpi=200, textfacecolor='white', textfacealpha=1.0, shortlabel=True, nocaption_withstring=[], timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None, timestampoffset=(0,0)):
        """Alias for savefig"""
        return self.savefig(outfile=outfile, 
                            categories=categories, 
                            figure=figure, 
                            nocaption=nocaption, 
                            fontsize=fontsize, 
                            boxalpha=boxalpha, 
                            d_category2color=d_category2color,
                            captionoffset=captionoffset, 
                            dpi=dpi, 
                            textfacecolor=textfacecolor, 
                            textfacealpha=textfacealpha, 
                            shortlabel=shortlabel, 
                            nocaption_withstring=nocaption_withstring, 
                            timestamp=timestamp, 
                            timestampcolor=timestampcolor, 
                            timestampfacecolor=timestampfacecolor,
                            timestampoffset=timestampoffset,
                            mutator=mutator)

    def savefig(self, outfile=None, categories=None, figure=1, nocaption=False, fontsize=10, boxalpha=0.25, d_category2color={'person':'green', 'vehicle':'blue', 'object':'red'}, captionoffset=(0,0), dpi=200, textfacecolor='white', textfacealpha=1.0, shortlabel=True, nocaption_withstring=[], timestamp=None, timestampcolor='black', timestampfacecolor='white', mutator=None, timestampoffset=(0,0)):
        """Save show() output to given file or return buffer without popping up a window"""
        fignum = figure if figure is not None else 1        
        self.show(categories=categories, figure=fignum, nocaption=nocaption, fontsize=fontsize, boxalpha=boxalpha, 
                  d_category2color=d_category2color, captionoffset=captionoffset, nowindow=True, textfacecolor=textfacecolor, 
                  textfacealpha=textfacealpha, shortlabel=shortlabel, nocaption_withstring=nocaption_withstring, timestamp=timestamp,
                  timestampcolor=timestampcolor, timestampfacecolor=timestampfacecolor, mutator=mutator, timestampoffset=timestampoffset)
        
        if outfile is None:
            buf = io.BytesIO()
            (W,H) = plt.figure(num=fignum).canvas.get_width_height()  # fast(ish)
            plt.figure(num=fignum).canvas.print_raw(buf)  # fast(ish)
            img = np.frombuffer(buf.getbuffer(), dtype=np.uint8).reshape((H, W, 4))
            if figure is None:
                vipy.show.close(plt.gcf().number)   # memory cleanup (useful for video annotation on last frame)
            return vipy.image.Image(array=img, colorspace='rgba')
        else:
            vipy.show.savefig(os.path.abspath(os.path.expanduser(outfile)), figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
            return outfile

        
    
class ImageDetection(Image, vipy.object.Detection):
    """vipy.image.ImageDetection class

    This class provides a representation of a `vipy.image.Image` with a single `vipy.object.Detection`.  This is useful for direct bounding box manipulations.

    This class inherits all methods of `vipy.image.Image` and `vipy.object.Detection` (and therefore `vipy.geometry.BoundingBox`).  

    Inheritance priority is for Image.  Overloaded methods such as rescale() or width() will transform or return values for the Image.

    Valid constructors include all provided by vipy.image.Image and BoundingBox coordinates

    ```python
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, width=100, height=100)
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, xmax=100, ymax=100)
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xcentroid=50, ycentroid=50, width=100, height=100)
    ```

    .. notes::
        - The inheritance resolution order will prefer the subclass methods for `vipy.image.Image`.  For example, the shape() method will return the image shape.
        - Use `vipy.image.DetectionImage` or `vipy.image.ImageDetection.detectionimage` cast if you prefer overloaded methods to resolve to bounding box manipulation..
        - All methods in this class will transform the pixels or the box independently.  The use case for this class is to manipulate boxes relative to the image for refinement (e.g. data augmentation).
        - If you want the pixels to be transformed along with the boxes, use the `vipy.image.ImageDetection.scene` method to cast this to a `vipy.image.Scene` object.
    """
    
    def __init__(self, filename=None, url=None, attributes=None, colorspace=None, array=None, 
                 xmin=None, xmax=None, ymin=None, ymax=None, width=None, height=None, 
                 xcentroid=None, ycentroid=None, category=None, xywh=None, bbox=None, id=True):

        # vipy.object.Detection inheritance
        vipy.object.Detection.__init__(self,
                                       xmin=xmin,
                                       ymin=ymin,
                                       width=width,
                                       height=height,
                                       xmax=xmax,
                                       ymax=ymax,
                                       xcentroid=xcentroid,
                                       ycentroid=ycentroid,
                                       xywh=xywh if xywh is not None else (bbox.xywh() if isinstance(bbox, BoundingBox) else None),                                       
                                       category=category,
                                       attributes=attributes,
                                       id=id)

        # vipy.image.Image class inheritance
        Image.__init__(self,
                       filename=filename,
                       url=url,
                       attributes=attributes,
                       array=array,
                       colorspace=colorspace)
        
    def __repr__(self):
        return str('<vipy.image.imagedetection: %s, %s>' % (Image.__repr__(self), vipy.object.Detection.__repr__(self)))
        
    def __eq__(self, other):
        """ImageDetection equality is defined as equivalent categories and boxes (not pixels)"""
        return self.boundingbox() == other.boundingbox() if isinstance(other, ImageDetection) else False

    def boundingbox(self):
        """Cast this object to a cloned `vipy.object.Detection` object"""
        return vipy.object.Detection.cast(self.clone())

    def scene(self):
        """Cast this object to a cloned `vipy.image.Scene` object"""        
        return vipy.image.Scene.cast(self.clone()).objects([self.boundingbox()])
    
    def crop(self):
        """Crop the image using the bounding box and return a `vipy.image.Image` for the cropped pixels"""
        return vipy.image.Image.cast(self.clone())._crop(self.boundingbox())

    def show(self):
        """Show this object by casting to `vipy.image.Scene`"""
        self.scene().show()
        return self

    def clone(self):
        """Clone the object, for finer control over clone, cast to `vipy.image.ImageDetection.scene`"""
        return copy.deepcopy(self)

    def detectionimage(self):
        """Cast the class to the base class vipy.image.DetectionImage so that bounding box methods have inheritance priority"""
        self.__class__ = vipy.image.DetectionImage
        return self

    def isinterior(self):
        """is the bounding box fully within the provided image?"""
        return super().isinterior(self.width(), self.height())

    
    
    
class DetectionImage(vipy.object.Detection, Image):
    """vipy.image.DetectionImage class

    This class provides a representation of a `vipy.image.Image` with a single `vipy.object.Detection`  This is useful for direct bounding box manipulations.

    This class inherits all methods of `vipy.image.Image` and `vipy.object.Detection` (and therefore `vipy.geometry.BoundingBox`).  

    Inheritance priority is for vipy.object.Detection.  Overloaded methods such as rescale() or width() will return values for the bounding box.

    Valid constructors include all provided by vipy.image.Image and BoundingBox coordinates

    ```python
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, width=100, height=100)
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, xmax=100, ymax=100)
    im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xcentroid=50, ycentroid=50, width=100, height=100)
    ```

    .. notes::
        - The inheritance resolution order will prefer the subclass methods for `vipy.object.Detection`.  For example, the shape() method will return the bounding box shape.
        - Use `vipy.image.ImageDetection` or `vipy.image.DetectionImage.imagedetection` cast if you prefer overloaded methods to resolve to image manipulation.
        - All methods in this class will transform the pixels or the box independently.  The use case for this class is to manipulate boxes relative to the image for refinement (e.g. data augmentation).
        - If you want the pixels to be transformed along with the boxes, use the `vipy.image.DetectionImage.scene` method to cast this to a `vipy.image.Scene` object.
    """
    
    def __init__(self, filename=None, url=None, attributes=None, colorspace=None, array=None, 
                 xmin=None, xmax=None, ymin=None, ymax=None, width=None, height=None, 
                 xcentroid=None, ycentroid=None, category=None, xywh=None, bbox=None, id=True):

        # vipy.image.Image class inheritance
        Image.__init__(self,
                       filename=filename,
                       url=url,
                       attributes=attributes,
                       array=array,
                       colorspace=colorspace)
        
        # vipy.object.Detection inheritance
        vipy.object.Detection.__init__(self,
                                       xmin=xmin,
                                       ymin=ymin,
                                       width=width,
                                       height=height,
                                       xmax=xmax,
                                       ymax=ymax,
                                       xcentroid=xcentroid,
                                       ycentroid=ycentroid,
                                       xywh=xywh if xywh is not None else (bbox.xywh() if isinstance(bbox, BoundingBox) else None),
                                       category=category,
                                       id=id)
                
    def __repr__(self):
        return str('<vipy.image.detectionimage: %s, %s>' % (vipy.object.Detection.__repr__(self), Image.__repr__(self)))
        
    def __eq__(self, other):
        """Equality is defined as equivalent categories and boxes (not pixels)"""
        return self.boundingbox() == other.boundingbox() if isinstance(other, ImageDetection) else False

    def boundingbox(self):
        """Cast this object to a cloned `vipy.object.Detection` object"""
        return vipy.object.Detection.cast(self.clone())

    def scene(self):
        """Cast this object to a cloned `vipy.image.Scene` object"""        
        return vipy.image.Scene.cast(self.clone()).objects([self.boundingbox()])
    
    def crop(self):
        """Crop the image using the bounding box and return a `vipy.image.Image` for the cropped pixels"""
        return vipy.image.Image.cast(self.clone())._crop(self.boundingbox())

    def show(self):
        """Show this object by casting to `vipy.image.Scene`"""
        self.scene().show()
        return self

    def clone(self):
        """Clone the object, for finer control over clone, cast to `vipy.image.ImageDetection.scene`"""
        return copy.deepcopy(self)

    def imagedetection(self):
        """Cast the class to the base class vipy.image.ImageDetection so that image methods have inheritance priority"""
        self.__class__ = vipy.image.ImageDetection
        return self

    def isinterior(self):
        """is the bounding box fully within the provided image?"""
        return super().isinterior(Image.width(self), Image.height(self))
    
    
def mutator_show_trackid(n_digits_in_trackid=5):
    """Mutate the image to show track ID with a fixed number of digits appended to the shortlabel as (####)"""
    return lambda im, k=None: (im.objectmap(lambda o: o.shortlabel('%s (%s)' % (o.shortlabel(), o.attributes['__trackid'][0:n_digits_in_trackid]))
                                            if o.hasattribute('__trackid') else o))

def mutator_show_jointlabel():
    """Deprecated"""
    return mutator_capitalize()

def mutator_show_trackindex():
    """Mutate the image to show track index appended to the shortlabel as (####)"""
    return lambda im, k=None: (im.objectmap(lambda o: o.shortlabel('%s (%d)' % (o.shortlabel(), int(o.attributes['__trackindex']))) if o.hasattribute('__trackindex') else o))

def mutator_show_trackonly():
    """Mutate the image to show track as a consistently colored box with no shortlabels"""
    f = mutator_show_trackindex()
    return lambda im, k=None, f=f: f(im).objectmap(lambda o: o.shortlabel('__%s' % o.shortlabel()))  # prepending __shortlabel will not show it, but will color boxes correctly
    
def mutator_show_userstring(strlist):
    """Mutate the image to show user supplied strings in the shortlabel.  The list be the same length oas the number of objects in the image.  This is not checked.  This is passed to show()"""
    assert isinstance(strlist, list), "Invalid input"
    return lambda im, k=None, strlist=strlist: im.objectmap([lambda o,s=s: o.shortlabel(s) for s in strlist])

def mutator_show_noun_only(nocaption=False):
    """Mutate the image to show the noun only.  
    
    Args:
        nocaption: [bool] If true, then do not display the caption, only consistently colored boxes for the noun. 
    
    ..note:: To color boxes by track rather than noun, use `vipy.image.mutator_show_trackonly`
    """
    return lambda im, k=None: (im.objectmap(lambda o: o.shortlabel('\n'.join([('__'+n if nocaption else n) for (n,v) in o.attributes['__noun verb']])) if o.hasattribute('__noun verb') else o))

def mutator_show_nounonly(nocaption=False):
    """Alias for `vipy.image.mutator_show_noun_only`"""
    return mutator_show_noun_only(nocaption=nocaption)

def mutator_show_verb_only():
    """Mutate the image to show the verb only"""
    return lambda im, k=None: (im.objectmap(lambda o: o.shortlabel('\n'.join([v for (n,v) in o.attributes['__noun verb']])) if o.hasattribute('__noun verb') else o))

def mutator_show_noun_or_verb():
    """Mutate the image to show the verb only if it is non-zero else noun"""
    return lambda im: (im.objectmap(lambda o: o.shortlabel('\n'.join([v if len(v)>0 else n for (n,v) in o.attributes['__noun verb']])) if o.hasattribute('__noun verb') else o))

def mutator_capitalize():
    """Mutate the image to show the shortlabel as 'Noun Verb1\nNoun Verb2'"""
    return lambda im, k=None: (im.objectmap(lambda o: o.shortlabel('\n'.join(['%s %s' % (n.capitalize(), v.capitalize()) for (n,v) in o.attributes['__noun verb']])) if o.hasattribute('__noun verb') else o))
    
def mutator_show_activityonly():
    return lambda im, k=None: im.objectmap(lambda o: o.shortlabel('') if (len(o.attributes['__noun verb']) == 1 and len(o.attributes['__noun verb'][0][1]) == 0) else o)

def mutator_show_trackindex_activityonly():
    """Mutate the image to show boxes colored by track index, and only show 'noun verb' captions"""
    f = mutator_show_trackindex()
    return lambda im, k=None, f=f: f(im).objectmap(lambda o: o.shortlabel('__%s' % o.shortlabel()) if (len(o.attributes['__noun verb']) == 1 and len(o.attributes['__noun verb'][0][1]) == 0) else o)

def mutator_show_trackindex_verbonly(confidence=True, significant_digits=2):
    """Mutate the image to show boxes colored by track index, and only show 'verb' captions with activity confidence, sorted in decreasing order"""
    f = mutator_show_trackindex()
    return lambda im, k=None, f=f: f(im).objectmap(lambda o: o.shortlabel('__%s' % o.shortlabel()) if (len(o.attributes['__noun verb']) == 1 and len(o.attributes['__noun verb'][0][1]) == 0) else o.shortlabel('\n'.join(['%s %s' % (v, ('(%1.2f)'%float(c)) if (confidence is True and c is not None) else '') for ((n,v),c) in sorted(zip(o.attributes['__noun verb'], o.attributes['__activityconf']), key=lambda x: float(x[1]), reverse=True)])))


def RandomImage(rows=None, cols=None):
    """Return a uniform random color `vipy.image.Image` of size (rows, cols)"""
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols
    return Image(array=np.uint8(255 * np.random.rand(rows, cols, 3)), colorspace='rgb')


def RandomImageDetection(rows=None, cols=None):
    """Return a uniform random color `vipy.image.ImageDetection` of size (rows, cols) with a random bounding box"""
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols
    return ImageDetection(array=np.uint8(255 * np.random.rand(rows, cols, 3)), colorspace='rgb', category='RandomImageDetection',
                          xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                          width=np.random.randint(16,cols), height=np.random.randint(16,rows))

def RandomScene(rows=None, cols=None, num_detections=8, num_keypoints=8, num_objects=None, url=None):
    """Return a uniform random color `vipy.image.Scene` of size (rows, cols) with a specified number of vipy.object.Object` objects"""    
    im = Scene(array=RandomImage(rows, cols).array(), category='RandomScene') if url is None else Scene(url=url, category='RandomScene')
    (rows, cols) = im.shape()
    objects = []
    if num_objects:
        (num_detection, num_keypoints) = (num_objects//2, num_objects//2)
    if num_detections:
        objects += [vipy.object.Detection('obj%04d' % k, xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16), width=np.random.randint(16,cols), height=np.random.randint(16,rows)) for k in range(num_detections)]
    if num_keypoints:
        objects += [vipy.object.Keypoint2d(category='kp%04d' % k, x=np.random.randint(0,cols - 16), y=np.random.randint(0,rows - 16), radius=np.random.randint(10,50)) for k in range(num_keypoints)]
    return im.objects(objects)
    

def owl():
    """Return a superb owl image for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/512px-Bubo_virginianus_06.jpg',
                 category='Nature',
                 objects=[vipy.object.Detection('Great Horned Owl', xmin=93, ymin=85, width=373, height=560)])

def squareowl():
    """Return a superb owl with no objects, cropped square at 512x512 resolution"""
    return owl().clear().centersquare().mindim(512)


def Owl():
    """Return superb owl centersquare, mindim=512"""
    return owl().centersquare().mindim(512)

def vehicles():
    """Return a highway scene with the four highest confidence vehicle detections for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/3/3e/I-80_Eastshore_Fwy.jpg',
                 category='Highway',
                 objects=[vipy.object.Detection(category="car", xywh=(473.0, 592.2, 92.4, 73.4)),
                          vipy.object.Detection(category="car", xywh=(1410.0, 756.1, 175.2, 147.3)),
                          vipy.object.Detection(category="car", xywh=(316.9, 640.1, 119.4, 119.5)),
                          vipy.object.Detection(category="car", xywh=(886.9, 892.9, 223.8, 196.6))]).mindim(512)

def people():
    """Return a crowd scene with the four highest confidence person detections for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/b/be/July_4_crowd_at_Vienna_Metro_station.jpg',
                 category='crowd',
                 objects=[vipy.object.Detection(category="person", xywh=(1.8, 1178.7, 574.1, 548.0)),
                          vipy.object.Detection(category="person", xywh=(1589.4, 828.3, 363.0, 887.7)),
                          vipy.object.Detection(category="person", xywh=(1902.9, 783.1, 250.8, 825.8)),
                          vipy.object.Detection(category="person", xywh=(228.2, 948.7, 546.8, 688.5))]).mindim(512)

def show(img, mindim=512, figure=1):
    """Fast visualization of a numpy array img
        
    ```python
    im = vipy.image.show(np.random.rand(16,16,3))
    ```

    """
    assert isnumpy(img)
    return vipy.image.Image(array=np.array(img).astype(np.float32), colorspace='float').mindim(mindim, interp='nearest').show(figure=figure)
