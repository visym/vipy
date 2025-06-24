import os
import PIL.Image
import PIL.ImageFilter
import PIL.ExifTags
import vipy.show
import vipy.globals
from vipy.globals import log, cache
from vipy.util import isnumpy, isurl, isimageurl, to_iterable, tolist,\
    fileext, tempimage, mat2gray, imwrite, imwritegray, mergedict, \
    tempjpg, filetail, isimagefile, remkdir, hasextension, truncate_string, \
    try_import, tolist, islistoflists, istupleoftuples, isstring, \
    islist, isnumber, isnumpyarray, string_to_pil_interpolation, toextension, \
    shortuuid, iswebp, has_image_extension, tocache, stringhash
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
from itertools import zip_longest
import functools


try:
    import ujson as json  # faster
except ImportError:        
    import json  # fastish
    

class Image():
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
    __slots__ = ('_filename', '_url', '_loader', '_array', '_colorspace', 'attributes')
    def __init__(self, filename=None, url=None, array=None, colorspace=None, attributes=None):
        # Private attributes
        self._loader = None     # function to load an image, set with loader() method
        self._array = None
        self._colorspace = None
        
        # Initialization
        self._filename = filename
        if url is not None:
            assert isinstance(url, str) and url.startswith(('http://', 'https://', 'scp://', 's3://'))  # faster than vipy.util.isurl()
        self._url = url
        if array is not None:
            assert isnumpy(array), 'Invalid Array - Type "%s" must be np.array()' % (str(type(array)))
        self.array(array)  # shallow copy

        # Colorspace guesses:
        if not colorspace:
            # Guess RGB colorspace if three channel uint8 if colorspace is not provided
            colorspace = 'rgb' if (self.isloaded() and self._array.ndim==3 and self._array.shape[2] == 3 and self._array.dtype == np.uint8) else colorspace

            # Guess LUM colorspace if three channel uint8 if colorspace is not provided
            colorspace = 'lum' if (self.isloaded() and (self._array.ndim==2 or (self._array.ndim==3 and self._array.shape[2] == 1)) and self._array.dtype == np.uint8) else colorspace
            
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
        return cls(filename=im._filename, url=im._url, array=im._array, colorspace=im._colorspace, attributes=im.attributes)


    @classmethod
    def from_dict(cls, d):
        d = {k.lstrip('_'):v for (k,v) in d.items()}  # prettyjson (remove "_" prefix to attributes)                
        return cls(filename=d['filename'] if 'filename' in d else None,
                   url=d['url'] if 'url' in d else None,
                   array=np.array(d['array'], dtype=np.uint8) if 'array' in d and d['array'] is not None else None,
                   colorspace=d['colorspace'] if 'colorspace' in d else None,
                   attributes=d['attributes'] if 'attributes' in d else None)
        
    @classmethod
    def from_uri(cls, uri):
        """Create an image object from an absolute file path or url"""
        assert vipy.util.isurl(uri) or vipy.util.isfile(uri), "invalid path"
        return cls(url=uri if vipy.util.isurl(uri) else None, filename=uri if vipy.util.isfile(uri) else None)            
    
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
        elif self.has_loader():
            strlist.append('loaded=False')
        if self.colorspace() == 'float':
            strlist.append('channels=%d' % self.channels())
        if self.filename() is not None:
            strlist.append('filename=%s' % self.filename())
        if self.hasurl():
            strlist.append('url=%s' % self.url())
        return str('<vipy.image.Image: %s>' % (', '.join(strlist)))

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
        
        Useful for debugging or sequential visualization in long fluent chains.
        """
        print(prefix+self.__repr__())
        if sleep is not None:
            assert sleep > 0, "Sleep must be a non-negative number of seconds"
            time.sleep(sleep)
        return self

    def exif(self, extended=False):
        """Return the EXIF meta-data in filename as a dictionary.  Included non-base EXIF data if extended=True.  Returns empty dictionary if no EXIF exists.  Triggers download but not load."""

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
            - The identity tiling is im.tile(im.width(), im.height(), overlaprows=0, overlapcols=0)
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
        return {k.lstrip('_'):getattr(self, k) for k in Image.__slots__}  # prettyjson (remove "_" prefix to attributes)                                    

    def json(self, encode=True):
        if not vipy.util.is_jsonable(self.attributes):
            raise ValueError('attributes dictionary contains non-json elements and cannot be serialized.  Try self.clear_attributes() or self.sanitize()')        
        d = {k:v for (k,v) in self.dict().items() if v is not None}  # filter empty
        if 'array' in d and d['array'] is not None:
            if self.hasfilename() or self.hasurl():
                log.warning('serializing pixel array to json is inefficient for large images.  Try self.flush() first, then reload the image from backing filename/url after json import')
            d['array'] = self._array.tolist()
        return json.dumps(d) if encode else d
        
    def loader(self, f, x=None):
        """Lambda function to load an unsupported image filename to a numpy array.
        
        This lambda function will be executed during load and the result will be stored in self._array
        """
        self._loader = (f, x if x is not None else self.filename()) if f is not None else None
        return self

    @staticmethod
    def bytes_array_loader(x):
        """Load from a bytes array"""
        return np.array(PIL.Image.open(io.BytesIO(x)))
    
    @staticmethod    
    def PIL_loader(x):
        """Load from a PIL image file object"""
        return np.array(x)

    def has_loader(self):
        return self._loader is not None

    
    def load(self, verbose=False):
        """Load image to cached private '_array' attribute.

        Args:
            verbose: [bool] If true, show additional useful printed output

        Returns:
            This `vipy.image.Image` object with the pixels loaded in self._array as a numpy array.

        .. note:: This loader supports any image file format supported by PIL.  A custom loader can be added using `vipy.image.Image.loader`.
        """
        try:
            # Return if previously loaded image
            if self._array is not None:
                return self

            # Download URL to filename 
            if self._url is not None and not self.hasfilename():
                self.download(verbose=verbose)

            # Load filename to numpy array
            if self._loader is not None:
                (f,x) = self._loader
                self._array = f(x)
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
            elif self.hasfilename() and hasextension(self._filename):
                raise ValueError('Non-standard image extensions require a custom loader')
            elif self.hasfilename():
                # Attempting to open it anyway, may be an image file without an extension. Cross your fingers ...
                self._array = np.array(PIL.Image.open(self._filename))  # RGB order!
            elif not self.hasfilename() and self.hasattribute('__shape'):
                # Loading a previously flushed buffer, load zeros so that we can display superclass objects
                self._array = np.zeros( self.getattribute('__shape') )
                self.delattribute('__shape')
            else:
                raise ValueError('image file not defined')
            
        except IOError:
            if verbose is True:
                log.error('IO error loading "%s" ' % self.filename())
            self._array = None
            raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if verbose is True:
                log.error('Load error for image "%s"' % self.filename())
            self._array = None
            raise

        return self

    def download(self, timeout=10, verbose=False, cached=False):
        """Download URL to filename provided by constructor, or to temp filename.

        Args:
            timeout: [int]  The timeout in seconds for an http or https connection attempt.  See also [urllib.request.urlopen](https://docs.python.org/3/library/urllib.request.html).
            verbose: [bool] If true, output more helpful message.
            cached: [bool] If true, use the cached previously downloaded file (if it exists)

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
                # There is a potential race condition here when downloading files with common names like "main.jpg", add a (repeatable, hashed) 3 character subdir (<=4096 subdirs for ext3, max ~32K)
                self._filename = os.path.join(remkdir(vipy.globals.cache()), stringhash(self._url, 3), filetail(self._url.split('?')[0]))  # preserve image filename from url
                self._filename = self._filename+'.jpg' if not has_image_extension(self._filename) else self._filename  # guess JPG for URLs with no file extension (e.g. php)
            elif isimageurl(self._url):
                self._filename = tempimage(fileext(self._url))
            else:
                self._filename = tempjpg()  # guess JPG for URLs with no file extension

        if cached and self.hasfilename():
            return self
            
        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if url_scheme in ['http', 'https']:
                vipy.downloader.download(self._url,
                                         self._filename,
                                         verbose=verbose,
                                         progress=False,
                                         timeout=timeout,
                                         sha1=self.getattribute('url_sha1'),
                                         username=self.getattribute('url_username'),
                                         password=self.getattribute('url_password'))
            elif url_scheme == 'file':
                shutil.copyfile(self._url, self._filename)
            elif url_scheme == 's3':
                raise NotImplementedError('see vipy.downloader.s3()')                
            else:
                raise NotImplementedError(
                    'Invalid URL scheme "%s" for URL "%s"' %
                    (url_scheme, self._url))

        except (httplib.BadStatusLine,
                urllib.error.URLError,
                urllib.error.HTTPError):
            if verbose is True:
                log.error('download failed for url "%s"' % self._url)
            self._array = None
            raise

        except IOError:
            if verbose:
                log.error('IO error downloading "%s" -> "%s" ' % (self.url(), self.filename()))
            self._array = None
            raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if verbose:
                log.error('load error for image "%s"' % self.filename())
            self._array = None
            raise

        return self

    def reload(self):
        """Flush the image buffer to force reloading from file or URL"""
        return self.clone(flush=True).load()

    def isloaded(self):
        """Return True if `vipy.image.Image.load` was successful in reading the image, or if the pixels are present in `vipy.image.Image.array`."""
        return self._array is not None

    def loaded(self):
        """Alias for `vipy.image.Image.isloaded`"""
        return self._array is not None

    def is_loaded(self):
        """Alias for `vipy.image.Image.isloaded`"""
        return self._array is not None
    
    def isdownloaded(self):
        """Does the filename returned from `vipy.image.Image.filename` exist, meaning that the url has been downloaded to a local file?"""
        return self._filename is not None and os.path.exists(self._filename)

    def is_downloaded(self):
        """Alias for ``vipy.image.Image.isdownloaded`"""
        return self.isdownloaded()
    
    def downloadif(self, timeout=10, verbose=False):
        """Download URL to filename if the filename has not already been downloaded"""
        return self.download(timeout=timeout, verbose=verbose, cached=True) if self.hasurl() else self

    def try_download(self, timeout=10, verbose=False):
        """Attempt to download URL to filename if the filename has not already been downloaded, return object on failure.  Check `vipy.image.Image.is_downloaded` on returned object for success"""
        try:
            return self.downloadif(timeout=timeout, verbose=verbose)
        except:
            return self

    def try_load(self):
        """Attempt to load an image, return the object on failure.  Check `vipy.image.Image.is_loaded` on returned object for success"""
        try:
            return self.load()
        except:
            return self
        
    def channels(self):
        """Return integer number of color channels"""
        return self.load().channels() if not self.isloaded() else (1 if self._array.ndim==2 else self._array.shape[2])

    def iscolor(self):
        """Color images are three channel or four channel with transparency, float32 or uint8"""
        return self.channels() == 3 or self.channels() == 4

    def istransparent(self):
        """Transparent images are four channel color images with transparency, float32 or uint8.  Return true if this image contains an alpha transparency channel"""
        return self.channels() == 4

    def blend(self, im, alpha):
        """alpha blend self and im in-place, such that self = alpha*self + (1-alpha)*im"""
        assert isinstance(im, Image)
        assert alpha >=0 and alpha <= 1
        assert self.colorspace() not in ['float','rgba','bgra'], "convert to rgb first"
        return self.load().map(lambda arr: np.uint8(alpha * arr + (1-alpha)*im.clone().load()._to_colorspace(self.colorspace()).resize_like(self).array()))
                
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
            self.colorspace(None)  # must be set with colorspace() after array() but before _to_colorspace()
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
        from torch import from_numpy;  # optional package pytorch not installed, run "pip install torch" (don't use try_import here, it's too slow)
        
        assert order in ['CHW', 'HWC', 'NCHW', 'NHWC']
        img = self.numpy() if self.array().ndim >= 3 else np.expand_dims(self.array(), 2)  # HxW -> HxWx1 
        
        if order in ['CHW']:
            assert img.ndim == 3, "invalid array"  
            img = img.transpose(2,0,1) # HxWxC -> CxHxW
        elif order in ['NCHW']:
            img = img.transpose(3,2,0,1) if img.ndim == 4 else np.expand_dims(img.transpose(2,0,1), 0)
        if order in ['NHWC']:
            img = img.transpose(3,0,1,2) if img.ndim == 4 else np.expand_dims(img, 0)
        return from_numpy(img)   # pip install torch

    
    @staticmethod
    def from_torch(x, order='CHW'):
        """Convert a 1xCxHxW, CxHxW or NxCxHxW torch tensor (or numpy array with torch channel order) to HxWxC numpy array, returns new `vipy.image.Image` with inferred colorspace corresponding to data type in x"""
        from torch import Tensor, is_tensor;  # optional package pytorch not installed, run "pip install torch" (don't use try_import here, it's too slow) 
        assert isinstance(x, Tensor) or isinstance(x, np.ndarray), "Invalid input type '%s'- must be torch.Tensor" % (str(type(x)))
        assert x.ndim == 4 or x.ndim == 3, "Torch tensor must be shape 1xCxHxW, CxHxW, or NxCxHxW"
        x = x.squeeze(0) if (x.ndim == 4 and x.shape[0] == 1) else x

        if order == 'CHW':
            x = x.permute(1,2,0).cpu().detach().float().numpy() if is_tensor(x) else np.copy(x).transpose(1,2,0)   # CxHxW -> HxWxC, copied            
        elif order == 'WHC':
            x = x.permute(1,0,2).cpu().detach().float().numpy() if is_tensor(x) else np.copy(x).transpose(1,0,2)   # WxHxC -> HxWxC, copied        
        elif order == 'HWC':
            x = x.cpu().detach().float().numpy() if is_tensor(x) else np.copy(x)  # HxWxC -> HxWxC, copied        
        elif order == 'NCHW':
            assert x.ndim == 4, "invalid shape"
            x = x.permute(2,3,1,0).cpu().detach().float().numpy()  # NxCxHxW -> HxWxCxN, copied        
        else:
            raise ValueError('unknown axis order "%s"' % order)

        img = x
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace  # assumed
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace
        return Image(array=img, colorspace=colorspace)

    @staticmethod
    def fromtorch(x, order='CHW'):
        """Alias for `vipy.image.Image.from_torch`"""
        return Image.from_torch(x, order)
    
    def unload(self):
        """Remove cached file and loaded array.  Note that this will delete the underlying file returned by filename() if there is a backing url, cleaning up cached files and forcing re-download"""
        if self.hasurl() and self.hasfilename():
            log.info('Removing "%s"'% self._filename)
            os.remove(self._filename)
            self._filename = None
        if self.isloaded():
            self.flush()
        return self

    def uncache(self):
        """Alias for `vipy.image.Image.unload`"""
        return self.unload()
    
    def filename(self, newfile=None):
        """Return or set image filename"""
        if newfile is None:
            return self._filename
        else:
            self._filename = newfile
            return self

    def clear_filename(self):
        """Remove the current filename from the object in-place and return the object"""        
        self._filename = None
        return self
    
    def url(self, url=None, username=None, password=None, sha1=None):
        """Image URL and URL download properties"""
        if url is not None:
            self._url = url  # this does not change anything else (e.g. the associated filename), better to use constructor 
        if username is not None:
            self.setattribute('url_username', username)
        if password is not None:
            self.setattribute('url_password', password)
        if sha1 is not None:
            self.setattribute('url_sha1', sha1)
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

    def set_attribute(self, key, value):
        """Set element self.attributes[key]=value"""
        if self.attributes is None:
            self.attributes = {key: value}
        else:
            self.attributes[key] = value
        return self
    
    def setattribute(self, key, value):
        return self.set_attribute(key, value)
        
    def setattributes(self, newattr):
        """Set many attributes at once by providing a dictionary to be merged with current attributes"""
        assert isinstance(newattr, dict), "New attributes must be dictionary"
        self.attributes.update(newattr)
        return self
    
    def getattribute(self, k):
        """Return the key k in the attributes dictionary (self.attributes) if present, else None"""        
        return self.get_attribute(k)

    def get_attribute(self, k):
        """Return the key k in the attributes dictionary (self.attributes) if present, else None"""        
        return self.attributes[k] if k in self.attributes else None        
    
    def clear_attributes(self):
        self.attributes = {}
        return self
    
    def hasattribute(self, key):
        return self.attributes is not None and key in self.attributes

    def delattribute(self, k):
        return self.del_attribute(k)
    
    def del_attribute(self, k):
        if k in self.attributes:
            self.attributes.pop(k)
        return self
        
    def delattributes(self, atts):
        for k in tolist(atts):
            self.delattribute(k)
        return self

    def append_attribute(self, key, value):
        """Append the value to attribute key, creating the key as an empty list if it does not exist"""
        if key not in self.attributes:
            self.attributes[key] = []
        self.attributes[key].append(value)
        return self
    
    def metadata(self, k=None):
        """Return metadata associated with this image, stored in the attributes dictionary"""
        return self.attributes if k is None else self.getattribute(k)
    
    def hasurl(self):
        """synonym for `vipy.image.has_url`"""
        return self.has_url()

    def has_url(self):
        """Return True if the image has a URL input source"""
        return self._url is not None
    
    def has_filename(self):
        """Return True if the image has a filename input source and this file exists"""
        return self._filename is not None and os.path.exists(self._filename)

    def hasfilename(self):
        """synonym for has_filename"""
        return self.has_filename()
    
    def clone(self, flushforward=False, flushbackward=False, flush=False, shallow=False, attributes=False, dereference=False):
        """Create deep copy of object, flushing the original buffer if requested and returning the cloned object.
        Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned 
        object which can be used for encoding and will be garbage collected.
        
            * flushforward: copy the object, and set the cloned object array() to None.  This flushes the video buffer for the clone, not the object
            * flushbackward:  copy the object, and set the object array() to None.  This flushes the video buffer for the object, not the clone.
            * flush:  set the object array() to None and clone the object.  This flushes the video buffer for both the clone and the object.
            * dereference: remove both the filename and URL (if present) in the cloned object, leaving only the buffer
        """
        if flush or (flushforward and flushbackward):
            self.flush()  # flushes buffer on object and clone
            im = copy.deepcopy(self)  # object and clone are flushed
        elif flushbackward:
            im = copy.deepcopy(self)  # propagates _array to clone
            self.flush()  # object flushed, clone not flushed
        elif flushforward:            
            array = self._array;
            self._array = None
            im = copy.deepcopy(self)   # does not propagate _array to clone
            self._array = array    # object not flushed
            im.flush()
        elif shallow:
            im = copy.copy(self)  # shallow copy
            im._array = np.asarray(self._array) if self._array is not None else None  # shared pixels            
        else:
            im = copy.deepcopy(self)
        if attributes:
            im.attributes = copy.deepcopy(self.attributes)
        if dereference:
            assert im._array is not None, "image buffer required"
            im._filename = None
            im._url = None
        return im

    def flush(self):
        """flush the image buffer in place, alias for self.clone(flush=True)"""        
        if not (self.hasfilename() or self.hasurl()):
            self.setattribute('__shape', (self.height(), self.width(), self.channels()))  # to load zeros
        self._array = None  # flushes buffer on object
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

    def pad(self, padwidth, padheight):
        """Alias for `vipy.image.Image.zeropad`"""
        return self.zeropad(padwidth, padheight)
    
    def zeropad(self, padwidth, padheight):
        """Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding,, and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding"""
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth, padwidth)
        if not isinstance(padheight, tuple):
            padheight = (padheight, padheight)
        if self.channels() > 1 or self._array.ndim == 3:
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
    def _to_colorspace(self, to):
        """Supported colorspaces are rgb, rgba, bgr, bgra, hsv, grey, lum, float"""
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
            self._to_colorspace(to)
        elif self.colorspace() in ['gray', 'grey']:
            img = self.load().array()  # single channel float32 [0,1]
            img = np.squeeze(img, axis=2) if img.ndim == 3 and img.shape[2] == 1 else img  # remove singleton channel                        
            self._array = np.array(PIL.Image.fromarray(255.0 * img, mode='F').convert('RGB'))  # float32 gray [0,1] -> float32 gray [0,255] -> uint8 RGB
            self.colorspace('rgb')
            self._to_colorspace(to)
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
                self._to_colorspace(to)
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
            self._to_colorspace(to)
        elif self.colorspace() == 'bgra':
            img = self.load().array()  # uint8 BGRA
            self._array = np.array(img)[:,:,::-1]  # uint8 BGRA -> uint8 ARGB
            self._array = self._array[:,:,[1,2,3,0]]  # uint8 ARGB -> uint8 RGBA
            self.colorspace('rgba')
            self._to_colorspace(to)
        elif self.colorspace() == 'hsv':
            img = self.load().array()  # uint8 HSV
            self._array = np.array(PIL.Image.fromarray(img, mode='HSV').convert('RGB'))  # uint8 HSV -> uint8 RGB
            self.colorspace('rgb')
            self._to_colorspace(to)
        elif self.colorspace() == 'float':
            img = self.load().array()  # float32
            if np.max(img) > 1 or np.min(img) < 0:
                #log.warning('Converting float image to "%s" will be rescaled with self.mat2gray() into the range float32 [0,1]' % to)
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
            self._to_colorspace(to)
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
        .. note:: This will transform only the pixels, not objects
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
        return self._to_colorspace('rgb')

    def color_transform(self, colorspace):
        """Transform the image buffer from the current `vipy.image.Image.colorspace` to the provided colorspace"""
        return self._to_colorspace(colorspace)
    
    def colorspace_like(self, im):
        """Convert the image buffer to have the same colorspace as the provided image"""
        assert isinstance(im, vipy.image.Image)
        return self._to_colorspace(im.colorspace())
    
    def rgba(self):
        """Convert the image buffer to four channel RGBA uint8 colorspace"""
        return self._to_colorspace('rgba')

    def hsv(self):
        """Convert the image buffer to three channel HSV uint8 colorspace"""
        return self._to_colorspace('hsv')

    def bgr(self):
        """Convert the image buffer to three channel BGR uint8 colorspace"""
        return self._to_colorspace('bgr')

    def bgra(self):
        """Convert the image buffer to four channel BGR uint8 colorspace"""
        return self._to_colorspace('bgra')

    def float(self):
        """Convert the image buffer to float32"""
        return self._to_colorspace('float')

    def greyscale(self):
        """Convert the image buffer to single channel grayscale float32 in range [0,1]"""
        return self._to_colorspace('gray')

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
        return self._to_colorspace('lum')

    def lum(self):
        """Alias for luminance()"""
        return self._to_colorspace('lum')

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
        #return self.array(vipy.math.gain(self.load()._array, np.float32(g))).colorspace('float') if g != 1 else self        
        #return self.array(np.float32(self.load()._array*g)).colorspace('float') if g != 1 else self  # numba not as fast anymore
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
        return self.array(gain*self.load().float().array() + bias).colorspace('float')

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
    
    def show(self, figure=1, nowindow=False, timestamp=None, mutator=None, theme='dark'):
        """Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object"""
        assert self.load().isloaded(), 'Image not loaded'
        timestampfacecolor = 'black' if theme=='dark' else 'white'
        timestampcolor = 'white' if theme=='dark' else 'black'
        im = self.clone() if not mutator else mutator(self.clone())        
        vipy.show.imshow(im.rgb().numpy(), fignum=figure, nowindow=nowindow, timestamp=timestamp, timestampfacecolor=timestampfacecolor, flush=True, timestampcolor=timestampcolor)
        return self

    def save(self, filename=None, quality=75):
        """Save the current image to a new filename and return the image object.  Resets edit history"""
        return self.filename(self.saveas(filename if filename	is not None else tempjpg(), quality=quality)).loader(None).flush_array()
        
        
    # Image export
    def pkl(self, pklfile=None):
        """save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains"""
        assert pklfile is not None or self.filename() is not None
        pklfile = pklfile if pklfile is not None else toextension(self.filename(), '.pkl')
        remkdir(vipy.util.filepath(pklfile))
        vipy.util.save(self, pklfile)
        return self

    def pklif(self, b, pklfile=None):
        """Save the object to the provided pickle file only if b=True. Useful for conditional intermediate saving in long fluent chains"""
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

    def annotate(self, timestamp=None, mutator=None, theme='dark'):
        """Change pixels of this image to include rendered annotation and return an image object"""
        # FIXME: for k in range(0,10): self.annotate().show(figure=k), this will result in cumulative figures
        return vipy.image.Image(array=self.savefig(timestamp=timestamp, theme=theme, mutator=mutator).rgb().array(), colorspace='rgb')

    def savefig(self, filename=None, figure=1, timestamp=None, theme='dark', mutator=None):
        """Save last figure output from self.show() with drawing overlays to provided filename and return filename"""
        self.show(figure=figure, nowindow=True, timestamp=timestamp, theme=theme, mutator=mutator)  # sets figure dimensions, does not display window
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

    def face_detection(self, mindim=256,  conf=0.2):
        """Detect faces in the scene, add as objects, return new scene with just faces
        
        Args:
            mindim [int]: The minimum dimension for downsampling the image for face detection.  Will be upsampled back to native resolution prior to return

        Returns
            A `vipy.image.Scene` object with all detected faces or the union of faces and all objects in self

        .. note:: This method uses a CPU-only pretrained face detector.  This is convenient, but slow.  See the heyvi package for optimized GPU batch processing for faster operation.
        """
        try_import('heyvi'); import heyvi; assert heyvi.version.is_at_least('0.3.28') 
        return heyvi.detection.FaceDetector()(Scene.cast(self.clone()).clear().mindim(mindim)).flush() 
    
    def person_detection(self, mindim=256, conf=0.2):
        """Detect only people in the scene, add as objects, return new scene with just people

        Args:
            mindim [int]: The minimum dimension for downsampling the image for person detection.  Will be upsampled back to native resolution prior to return
            conf [float]: A real value between [0,1] of the minimum confidence for person detection

        Returns
            A `vipy.image.Scene` object with all detected people or the union of people and all objects in self
        
        .. note:: This method uses a CPU-only pretrained person detector.  This is convenient, but slow.  See the heyvi package for optimized GPU batch processing for faster operation.
        """
        try_import('heyvi'); import heyvi; assert heyvi.version.is_at_least('0.3.28')
        return heyvi.detection.ObjectDetector()(Scene.cast(self.clone()).clear().mindim(mindim), conf=conf, objects=['person']).flush()

    def face_blur(self, radius=4, mindim=256):
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
        im = self.face_detection(mindim=mindim)  # only faces
        return im.setattribute('face_blur', [o.int().json(encode=False) for o in im.objects()]).blurmask(radius=radius).downcast()

    def face_pixelize(self, radius=7, mindim=256):
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
        im = self.face_detection(mindim=mindim)          
        return im.setattribute('face_pixelize', [o.int().json(encode=False) for o in im.objects()]).pixelize(radius=radius).downcast()


    def viewport(self):
        """Return the bounding box of the current loaded pixels in the original filename/url/buffer.

        This reverses the chain of geometric transformations applied to the original image to recover the bounding box of the pixels in array().

        This is useful to specify a region of a larger image that was zoomed in for processing.
        
        To show this viewport as a bounding box:

        >>> im = vipy.image.vehicles().centercrop(100,100)
        >>> viewport = vipy.object.Detection.cast(im.viewport())
        >>> im.flush().append(viewport).show()
        """
        bb = self.imagebox()
        if self._history() is not None:
            for (f,kwargs) in reversed(self._history()):
                getattr(bb,f)(**kwargs)
        return bb

    def padcrop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects"""
        dx = int(max(0, max(0-bbox.xmin(), bbox.xmax()-self.width())))
        dy = int(max(0, max(0-bbox.ymin(), bbox.ymax()-self.height())))
        return self.zeropad(dx,dy)._crop(bbox.translate(dx=dx, dy=dy))
    
    def recenter(self, p):
        """Recenter the image so that point p=(x=col, y=row) in the current image is in the middle of the new image, zeropad to (width, height).  
           This is useful to implement a 'saccade', under the small angle assumption, where a rotation is approximated by a translation
        """        
        return self.padcrop(self.imagebox().centroid(p))

    
class Labeled(Image):
    """A labeled image is an image that contains some form of annotation.  This class is useful for identifying if an image has any annotatation at all or is completely unlabeled.

    >>> im = vipy.image.owl()
    >>> assert isinstance(im, vipy.image.Labeled)
    >>> im = vipy.image.RandomImage()
    >>> assert not isinstance(im, vipy.image.Labeled)    

    The specific form of annotation may be `vipy.image.ImageCategory`, `vipy.image.TaggedImage` or `vipy.image.Scene`, but all are `vipy.image.Labeled` 
    """
    pass


class ImageCategory(Labeled):
    """vipy ImageCategory class

    This class provides a representation of a vipy.image.Image with a category label. 

    Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label') and optional confidence

    ```python
    im = vipy.image.ImageCategory(filename='/path/to/dog_image.ext', category='dog')
    im = vipy.image.ImageCategory(url='http://path/to/dog_image.ext', category='dog')
    im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog')
    ```
    """

    __slots__ = ('_filename', '_url', '_loader', '_array', '_colorspace', 'attributes')    
    def __init__(self, filename=None, url=None, category=None, label=None, attributes=None, array=None, colorspace=None, confidence=None):
        # Image class inheritance
        super().__init__(filename=filename,
                         url=url,
                         attributes=attributes,
                         array=array,
                         colorspace=colorspace)

        self.set_attribute('category', category)
        if confidence is not None:
            self.set_attribute('confidence', float(confidence))

    def __repr__(self):
        fields = ['category=%s' % str(self.category())]
        fields +=  ['confidence=%1.3f' % self.confidence()] if self.confidence() is not None else []
        return super().__repr__().replace('vipy.image.Image', 'vipy.image.ImageCategory').replace('>', ', %s>' % ','.join(fields))

    def __eq__(self, other):
        return self.category() == other.category() if isinstance(other, ImageCategory) else False

    def __ne__(self, other):
        return self.category() != other.category() if isinstance(other, ImageCategory) else True

    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        return cls(filename=d['filename'] if 'filename' in d else None,
                   url=d['url'] if 'url' in d else None,
                   category=None,  # will be in attribute
                   tags=None,      # will be in attributes
                   confidence=None, 
                   attributes=d['attributes'] if 'attributes' in d else None,
                   colorspace=d['colorspace'] if 'colorspace' in d else None,
                   array=np.array(d['array'], dtype=np.uint8) if 'array' in d and d['array'] is not None else None)
    
    def new_category(self, c):
        return self.set_attribute('category', c)

    def clear_category(self):
        if 'category' in self.attributes:
            del self.attributes['category']
        return self
    
    def category(self):
        return self.attributes['category'] if 'category' in self.attributes else None  # self.attributes.get('category') 

    def confidence(self):
        return self.get_attribute('confidence')        

    def tags(self, tags=None):
        if tags is not None:
            return self.set_attribute('category', tolist(tags)[0])                
        return (self.category(), ) if self.category() is not None else ()

    
class TaggedImage(Labeled):
    """vipy.image.TaggedImage class

    This class provides a representation of a vipy.image.Image with one or more tags.

    Valid constructors include all provided by vipy.image.Image with additional labels that provide ground truth for the content of the image. 

    ```python
    im = vipy.image.TaggedImage(filename='/path/to/dog.jpg', tags={'dog','canine'})
    ```
    """
    __slots__ = ('_filename', '_url', '_loader', '_array', '_colorspace', 'attributes')        
    def __init__(self, filename=None, url=None, attributes=None, array=None, colorspace=None, tags=None, category=None, confidence=None, caption=None):
        super().__init__(filename=filename,
                         url=url,
                         attributes=attributes,
                         array=array,
                         colorspace=colorspace)
        
        tags = ([category] if category is not None else []) + (tolist(tags) if tags is not None else [])
        if len(tags) > 0:
            self.set_attribute('tags', tags)
        if caption is not None:
            self.captions(caption)
            
    def __repr__(self):
        fields  = ['category=%s' % self.category()] if len(self.tags())==1 else []
        fields += ['caption=%s' % truncate_string(self.caption(), 40)] if self.caption() is not None else []        
        fields +=  ['confidence=%1.3f' % self.confidence()] if len(self.tags())==1 and self.confidence() is not None else []
        fields +=  ['tags=%s' % truncate_string(str(self.tags()), 40)] if len(self.tags())>1 else []
        return super().__repr__().replace('vipy.image.Image', 'vipy.image.TaggedImage').replace('>', ', %s>' % ', '.join(fields))
        

    @classmethod
    def from_json(cls, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        return cls(filename=d['filename'] if 'filename' in d else None,
                   url=d['url'] if 'url' in d else None,
                   category=None,  # will be in attribute
                   tags=None,      # will be in attributes
                   attributes=d['attributes'] if 'attributes' in d else None,
                   colorspace=d['colorspace'] if 'colorspace' in d else None,
                   array=np.array(d['array'], dtype=np.uint8) if 'array' in d and d['array'] is not None else None)

    def category(self):
        return self.attributes['tags'][0] if 'tags' in self.attributes else None

    def new_category(self, c):
        self.attributes['tags'] = [c]
        self.del_attribute('confidences')
        return self
        
    def confidence(self, tag=None, default=None):
        t = tag if tag is not None else self.category()
        return self.get_attribute('confidences')[t] if self.hasattribute('confidences') and t in self.attributes['confidences'] else default

    def has_tag(self, t):
        return t in self.tags()
    
    def tags(self, tags=None):
        if tags is not None:
            return self.set_attribute('tags', tolist(tags))        
        return self.attributes['tags'] if 'tags' in self.attributes else []
    
    def add_tag(self, tag, confidence=None):
        self.append_attribute('tags', tag)
        if confidence is not None:
            if not self.hasattribute('confidences'):
                self.set_attribute('confidences', {})
            self.attributes['confidences'][tag] = confidence
        return self

    def add_caption(self, caption):
        self.append_attribute('captions', caption)
        return self
    
    def caption(self):
        return self.get_attribute('captions')[0] if self.hasattribute('captions') else None
    
    def captions(self, captions=None):
        if captions is not None:
            return self.set_attribute('captions', tolist(captions))
        return self.get_attribute('captions') if self.hasattribute('captions') else []
    
    def add_tags(self, tags, confidences=[]):
        for (t,c) in zip_longest(tags, confidences):
            self.add_tag(t, c)
        return self

    def clear_tags(self):        
        self.set_attribute('tags',[])
        if 'confidences' in self.attributes:
            del self.attributes['confidences']
        return self
    
    def add_soft_tags(self, soft_tags):
        """Soft tags are a list of (tag, confidence) tuples"""
        for (t,c) in soft_tags:
            self.add_tag(t, c)
        return self

    def add_soft_tag(self, soft_tag):
        """A soft tag is a tuple of (tag, confidence)"""
        return self.add_tag(*soft_tag)
    
    def soft_tags(self):
        """Soft tags are a list of (tag, confidence) tuples.  Will return only those tags with associated confidences.  Will return empty tuple if there are tags but no confidences"""
        return tuple((t, self.attributes['confidences'].get(t)) for t in self.tags() if 'confidences' in self.attributes and self.attributes['confidences'].get(t) is not None)

    def has_soft_tags(self):
        """Return true if there exist a confidence for any tag"""
        return len(self.soft_tags())>0

    
class Scene(TaggedImage):
    """vipy.image.Scene class

    This class provides a representation of a vipy.image.TaggedImage with one or more vipy.object.Object.  The goal of this class is to provide a unified representation for all objects in a scene.

    Valid constructors include all provided by vipy.image.Image() and vipy.image.ImageCategory() with the additional kwarg 'objects', which is a list of vipy.object.Object()

    ```python
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='city', objects=[vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='city').objects([vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels='face', xywh=[0,0,100,100])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels='face', xywh=[[0,0,100,100], [100,100,200,200]])
    im = vipy.image.Scene(filename='/path/to/city_image.jpg', category='office', boxlabels=['face', 'desk'] xywh=[[0,0,100,100], [200,200,300,300]])
    ```

    """
    __slots__ = ('_filename', '_url', '_loader', '_array', '_colorspace', 'attributes', '_objectlist')
    
    def __init__(self, filename=None, url=None, category=None, attributes=None, objects=None, xywh=None, boxlabels=None, array=None, colorspace=None, tags=None):
        super().__init__(filename=filename, url=url, attributes=attributes, tags=tags, category=category, array=array, colorspace=colorspace)  
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
                detlist = [d.new_category(label) for d in detlist]
            elif (isinstance(boxlabels, tuple) or islist(boxlabels)) and len(boxlabels) == len(xywh):
                detlist = [d.new_category(label) for (d,label) in zip(detlist, boxlabels)]
            else:
                raise ValueError("Invalid boxlabels list - len(boxlabels) must be len(xywh) with corresponding labels for each xywh box  [label1, label2, ...]")

        self._objectlist = self._objectlist + detlist

        
    @classmethod
    def cast(cls, im):
        assert isinstance(im, vipy.image.Image), "Invalid input - must be derived from vipy.image.Image"
        if im.__class__ != vipy.image.Scene:
            return cls(filename=im._filename, url=im._url, attributes=im.attributes, array=im._array, colorspace=im._colorspace).loader(*im._loader)
        return im
    
    @classmethod
    def from_json(obj, s):
        im = super().from_json(s)
        im.__class__ = vipy.image.Scene
        d = {k.lstrip('_'):v for (k,v) in (json.loads(s) if not isinstance(s, dict) else s).items()}  # prettyjson (remove "_" prefix to attributes)
        if 'objectlist' in d and isinstance(d['objectlist'], dict):
            # Version 1.15.1: expanded serialization to support multiple object types
            im._objectlist = [vipy.object.Detection.from_json(s) for s in d['objectlist']['Detection']] if 'Detection' in  d['objectlist'] else []
            im._objectlist += [vipy.object.Keypoint2d.from_json(s) for s in d['objectlist']['Keypoint2d']] if 'Keypoint2d' in  d['objectlist'] else []
        else:
            # Legacy support: 1.14.4
            im._objectlist = [vipy.object.Detection.from_json(s) for s in d['objectlist']]            
        return im

    def __json__(self):
        """Serialization method for json package"""
        return self.json(encode=True)

    def num_objects(self):
        return len(self._objectlist)
    
    def json(self, encode=True):
        d = {k.lstrip('_'):getattr(self, k) for k in Scene.__slots__ if getattr(self, k) is not None}  # prettyjson (remove "_" prefix to attributes)          
        d['objectlist'] = {'Detection': [bb.json(encode=False) for bb in self._objectlist if isinstance(bb, vipy.object.Detection)],
                           'Keypoint2d': [p.json(encode=False) for p in self._objectlist if isinstance(p, vipy.object.Keypoint2d)]}
        d['objectlist'] = {k:v for (k,v) in  d['objectlist'].items() if len(v) > 0}  # cleanup empty lists
        if 'attributes' in d and len(d['attributes'])==0:  # cleanup empty attributes
            del d['attributes']  # will be recreated in from_json
        if 'array' in d and d['array'] is not None:
            if self.hasfilename() or self.hasurl():
                log.warning('serializing pixel array to json is inefficient for large images.  Try self.flush() or self.save(), then reload the image from backing filename/url after json import')            
            d['array'] = self._array.tolist()        
        return json.dumps(d) if encode else d

        
    def __eq__(self, other):
        """Scene equality requires equality of all objects in the scene, assumes a total order of objects"""
        return isinstance(other, Scene) and len(self)==len(other) and all([obj1 == obj2 for (obj1, obj2) in zip(self, other)])

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        elif self.has_loader():
            strlist.append('loaded=False')
        if self.filename() is not None:
            strlist.append('filename=%s' % (self.filename()))
        if self.hasurl():
            strlist.append('url=%s' % self.url())
        if len(self.image_tags())==1:
            strlist += ['category=%s' % truncate_string(str(self.category()), 40)]
        elif len(self.image_tags())>1:
            strlist += ['tags=%s' % truncate_string(str(self.image_tags()), 40)]            
        if len(self.objects()) > 0:
            strlist.append('objects=%d' % len(self.objects()))
            
        return str('<vipy.image.Scene: %s>' % (', '.join(strlist)))

    def __len__(self):
        """The length of a scene is equal to the number of objects present in the scene"""
        return len(self._objectlist)

    def __iter__(self):
        """Iterate over each ImageDetection() in the scene"""
        for (k, im) in enumerate(self._objectlist):
            yield self.__getitem__(k)

    def __getitem__(self, k):
        """Return the kth object in the scene as a `vipy.image.Scene` object """
        assert isinstance(k, int), "Indexing by object in scene must be integer"
        return self.clone(shallow=True).objects([self._objectlist[k].clone()])

    def image_tags(self, tags=None):
        """Return the image level tags of the scene"""
        return super().tags(tags)
    
    def tags(self, tags=None):
        """Return the image level and object level tags of the scene"""        
        if tags is not None:
            return super().tags(tags) 
        return super().tags() + self.object_tags()
    
    def load(self, verbose=False):
        super().load(verbose=verbose)
        if self.is_loaded() and self.num_objects() > 0 and any(o.has_normalized_coordinates() for o in self.objects()):
            # Normalized coordinates are in the range [0,1] relative to the (height, width) which is not known until load()
            self.objectmap(lambda o: o.scale_x(self.array().shape[1]).scale_y(self.array().shape[0]).del_attribute('normalized_coordinates') if o.has_normalized_coordinates() else o)
        return self
    
    def split(self):
        """Split a scene with K objects into a list of K `vipy.image.Scene` objects, each with one object in the scene.
        
        .. note:: The pixel buffer is shared between each split.  Use [im.clone() for im in self.split()] for an explicit copy.
        """
        return list(self)

    def split_and_recenter(self):
        """Split a scene with K objects into a list of K `vipy.image.Scene` objects, each with one object in the scene, with the scene centered on the object with zeropadding
        
        .. note:: The pixel buffer is shared between each split.  Use [im.clone() for im in self.split()] for an explicit copy.
        """
        return [im.clone().recenter(im.boundingbox().centroid()) for im in self.split()]
    
    def append_object(self, imdet):
        """Append the provided vipy.object.Detection object to the scene object list"""
        assert isinstance(imdet, vipy.object.Object), "Invalid input"
        self._objectlist.append(imdet)
        return self

    def add_object(self, imdet):
        """Alias for append"""        
        return self.append_object(imdet)
    
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
        assert isinstance(other, Image)
        if isinstance(other, Scene):
            self.objects(self.objects()+other.objects())
        return self

    def __or__(self, other):
        super().__or__(other)
        return self.union(other)
    
    def uncrop(self, bb, shape):
        """Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)"""
        super().uncrop(bb, shape)
        return self.objectmap(lambda o: o.translate(bb.xmin(), bb.ymin()))
        
    def clear(self):
        """Remove all objects from this scene."""
        return self.objects([])
    
    def boundingbox(self):
        """The boundingbox of a scene is the union of all object bounding boxes, or None if there are no objects.  Load to compensate for normalized coordinates"""
        boxes = [vipy.geometry.BoundingBox.cast(bb) for bb in self.load().objects()]
        bb = boxes[0].clone() if len(boxes) >= 1 else None
        return bb.union(boxes[1:]) if len(boxes) >= 2 else bb

    def object_tags(self):
        """Return list of unique object tags in scene"""
        return list(dict.fromkeys([t for o in self.objects() for t in o.tags()]))
    
    # Spatial transformation
    def _history(self, func=None, **kwargs):
        """The undo history for flush. This is useful for remote processing of images at lower resolutions and square crops without passing around the image buffer"""
        if func is not None:
            self.append_attribute('_history', (func, kwargs))
            return self
        return self.getattribute('_history')

    def flush_array(self):
        return self.flush(undo_history=False)
    
    def flush(self, undo_history=True):
        """Free the image buffer, and undo all of the object transformations to restore alignment with the reference image filename/url"""
        if undo_history and self._history() is not None:
            for (f,kwargs) in reversed(self._history()):
                self.objectmap(lambda o: getattr(o,f)(**kwargs))  # undo
        self.delattribute('_history')
        return super().flush()
    
    def imclip(self):
        """Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image"""
        self._objectlist = [o.imclip(self.numpy()) for o in self._objectlist if o.hasoverlap(self.numpy())]
        return self

    def rescale(self, scale=1, interp='bilinear'):
        """Rescale image buffer and all bounding boxes - Not idempotent"""
        self = super().rescale(scale, interp=interp)
        self._objectlist = [bb.rescale(scale) for bb in self._objectlist]
        self._history('rescale', s=1/scale)
        return self

    def resize(self, cols=None, rows=None, height=None, width=None, interp='bilinear'):
        """Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly.  If cols or rows is None, then scale isotropically.  cols is a synonym for width, rows is a synonym for height"""
        assert not (cols is not None and width is not None), "Define either width or cols"
        assert not (rows is not None and height is not None), "Define either height or rows"
        rows = rows if height is None else height
        cols = cols if width is None else width        
        assert cols is not None or rows is not None, "Invalid input"
        
        sx = (float(cols) / self.width()) if cols is not None else None
        sy = (float(rows) / self.height()) if rows is not None else None
        sx = sy if sx is None else sx
        sy = sx if sy is None else sy        
        self._objectlist = [bb.scale_x(sx).scale_y(sy) for bb in self._objectlist]
        self._history('scale_x', s=1/sx)._history('scale_y', s=1/sy)
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
        self._history('translate', dx=dx, dy=dy)
        return self
    
    def fliplr(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self._objectlist = [bb.fliplr(self.numpy()) for bb in self._objectlist]
        self._history('fliplr', width=self.width())
        self = super().fliplr()
        return self

    def flipud(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self._objectlist = [bb.flipud(self.numpy()) for bb in self._objectlist]
        self._history('flipud', height=self.height())        
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
        self._history('translate', dx=-dx, dy=-dy)
        return self

    def meanpad(self, padwidth, padheight, mu=None):
        """Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super().meanpad(padwidth, padheight, mu=mu)
        dx = padwidth[0] if isinstance(padwidth, tuple) and len(padwidth) == 2 else padwidth
        dy = padheight[0] if isinstance(padheight, tuple) and len(padheight) == 2 else padheight
        self._objectlist = [bb.translate(dx, dy) for bb in self._objectlist]
        self._history('translate', dx=-dx, dy=-dy)
        return self

    def rot90cw(self):
        """Rotate the scene 90 degrees clockwise, and update objects"""
        (H,W) = self.shape()        
        self.array(np.rot90(self.numpy(), 3))
        self._objectlist = [bb.rot90cw(H, W) for bb in self._objectlist]
        self._history('rot90ccw', H=W, W=H)                
        return self

    def rot90ccw(self):
        """Rotate the scene 90 degrees counterclockwise, and update objects"""
        (H,W) = self.shape()
        self.array(np.rot90(self.numpy(), 1))
        self._objectlist = [bb.rot90ccw(H, W) for bb in self._objectlist]
        self._history('rot90cw', H=W, W=H)                        
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
        self._history('translate', dx=dx, dy=dy)                        
        return self

    def objectcrop(self, dilate=1.0):
        """Crop image using the `vipy.image.Scene.boundingbox` with dilation factor.  Crop will be zeropadded if outside the image rectangle."""
        bb = self.boundingbox()
        return self.padcrop(bb.dilate(dilate)) if bb is not None else self

    def objectsquare(self, dilate=1.0):
        """Crop image using the `vipy.image.Scene.boundingbox` with dilation factor, setting to maxsquare prior to crop.  Crop will be zeropadded if outside the image rectangle."""
        bb = self.boundingbox()
        return self.padcrop(bb.dilate(dilate).maxsquare()) if bb is not None else self        
    
    def centercrop(self, height, width):
        """Crop image of size (height x width) in the center, keeping the image centroid constant"""
        return self.crop(BoundingBox(xcentroid=float(self.width() / 2.0), ycentroid=float(self.height() / 2.0), width=int(width), height=int(height)))

    def cornercrop(self, height, width):
        """Crop image of size (height x width) from the upper left corner, returning valid pixels only"""
        return self.crop(BoundingBox(xmin=0, ymin=0, width=int(width), height=int(height)))
    
    def padcrop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects"""
        bbox = bbox.clone()
        dx = int(max(0, max(0-bbox.xmin(), bbox.xmax()-self.width())))
        dy = int(max(0, max(0-bbox.ymin(), bbox.ymax()-self.height())))
        self.zeropad(dx,dy)._crop(bbox.translate(dx=dx, dy=dy))
        (dx, dy) = (bbox.xmin(), bbox.ymin())
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist] # after crop        
        self._history('translate', dx=dx, dy=dy)                                
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
        for o in self._objectlist:
            if isinstance(o, vipy.geometry.BoundingBox) and o.hasoverlap(immask):
                bbm = o.clone().imclip(self.numpy()).int()
                immask[bbm.ymin():bbm.ymax(), bbm.xmin():bbm.xmax()] = 1
            if isinstance(o, vipy.geometry.Point2d) and o.boundingbox().hasoverlap(immask):
                mask = vipy.calibration.circle(o.x, o.y, o.r, W, H)
                immask[mask>0] = 1
        return immask

    def binarymask(self):
        """Alias for rectangular_mask with in-place update"""
        mask = self.rectangular_mask() if self.channels() == 1 else np.expand_dims(self.rectangular_mask(), axis=2)
        img = self.numpy()
        img[:] = mask[:]  # in-place update
        return self
        
    def bgmask(self):
        """Set all pixels outside object bounding boxes to zero"""
        mask = self.rectangular_mask() if self.channels() == 1 else np.expand_dims(self.rectangular_mask(), axis=2)
        img = self.numpy()
        img[:] = np.multiply(img, mask)  # in-place update
        return self  

    def fgmask(self):
        """Set all pixels inside object bounding boxes to zero"""
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
    
        
    def show(self, categories=None, figure=1, nocaption=False, nocaption_withstring=[], fontsize=10, boxalpha=0.15, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, captionoffset=(3,-18), nowindow=False, shortlabel=None, timestamp=None, mutator=None, timestampoffset=(0,0), theme='dark'):
        """Show scene detection 

        Args:
           - categories: [list]  List of category names in the scene to show
           - fontsize: [int] or [str]: Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size
           - figure: [int|str] Figure number or title, show the image in the provided figure=int numbered window
           - nocaption: [bool]  Show or do not show the text caption in the upper left of the box 
           - nocaption_withstring: [list]:  Do not show captions for those object categories containing any of the strings in the provided list
           - boxalpha (float, [0,1]):  Set the text box background to be semi-transparent with an alpha
           - d_category2color (dict):  Define a dictionary of required mapping of specific category() to box colors.  Non-specified categories are assigned a random named color from vipy.show.colorlist()
           - caption_offset (int, int): The relative position of the caption to the upper right corner of the box.
           - nowindow (bool):  Display or not display the image, used by `vipy.image.Scene.annotation`
           - shortlabel (dict):  An optional dictionary mapping category names to short names easier to display 
           - mutator (lambda):  A lambda function with signature lambda im: f(im) which will modify this image prior to show.  Useful for changing labels on the fly
           - timestampoffset (tuple): (x,y) coordinate offsets to shift the upper left corner timestamp
           - theme [str]: If 'dark' use dark mode, if 'light' use light mode to visualize captions with high contrast dark or light foregrounds 
        """
        colors = vipy.show.colorlist(theme)
        all_colors = vipy.show.colorlist()        
        textfacecolor = 'black' if theme=='dark' else 'white'
        timestampcolor = 'white' if theme=='dark'  else 'black'
        timestampfacecolor = 'black' if theme=='dark' else 'white'        
        textfacealpha = 0.8 if theme=='dark' else 0.85
        
        im = self.clone() if not mutator else mutator(self.clone())
        imdisplay = im.rgb() if im.colorspace() != 'rgb' else im.load()  # convert to RGB for show() if necessary
        
        valid_objects = [obj.clone() for obj in imdisplay.objects() if categories is None or obj.category() in tolist(categories)]  # Objects with valid category
        valid_objects = [obj.imclip(self.numpy()) for obj in valid_objects if obj.hasoverlap(self.numpy())]  # Objects within image rectangle
        valid_objects = [obj.new_category(shortlabel[obj.category()]) for obj in valid_objects] if shortlabel else valid_objects  # Display name as shortlabel?
        d_det_category_to_color = {d.category():colors[int(hashlib.sha1(str(d.category()).encode('utf-8')).hexdigest(), 16) % len(colors)] for d in valid_objects if isinstance(d, vipy.object.Detection)}
        d_kp_category_to_color = {d.category():all_colors[int(hashlib.sha1(str(d.category()).encode('utf-8')).hexdigest(), 16) % len(all_colors)] for d in valid_objects if isinstance(d, vipy.object.Keypoint2d)}        
        d_category_to_color = mergedict(d_kp_category_to_color, d_det_category_to_color, d_category2color)
        
        object_color = [d_category_to_color[d.category()] for d in valid_objects]                
        valid_objects  = [d if not any([c in d.category() for c in tolist(nocaption_withstring)]) else d.nocategory() for d in valid_objects]  # Objects requested to show without caption

        fontsize_scaled = float(fontsize.split(':')[0])*(min(imdisplay.shape())/640.0) if isstring(fontsize) else fontsize
        vipy.show.imobjects(imdisplay._array, valid_objects, bordercolor=object_color, textcolor=object_color, fignum=figure, do_caption=(nocaption==False), facealpha=boxalpha, fontsize=fontsize_scaled,
                            captionoffset=captionoffset, nowindow=nowindow, textfacecolor=textfacecolor, textfacealpha=textfacealpha, timestamp=timestamp,
                            timestampcolor=timestampcolor, timestampfacecolor=timestampfacecolor, timestampoffset=timestampoffset)
        return self

    def annotate(self, outfile=None, categories=None, figure=1, nocaption=False, fontsize=10, boxalpha=0.15, d_category2color={'person':'green', 'vehicle':'blue', 'object':'red'}, captionoffset=(3,-18), dpi=200, shortlabel=None, nocaption_withstring=[], timestamp=None, mutator=None, timestampoffset=(0,0), theme='dark'):
        """Alias for `vipy.image.Scene.savefig"""
        return self.savefig(outfile=outfile, 
                            categories=categories, 
                            figure=figure, 
                            nocaption=nocaption, 
                            fontsize=fontsize, 
                            boxalpha=boxalpha, 
                            d_category2color=d_category2color,
                            captionoffset=captionoffset, 
                            dpi=dpi, 
                            shortlabel=shortlabel, 
                            nocaption_withstring=nocaption_withstring, 
                            timestamp=timestamp,
                            theme=theme,
                            timestampoffset=timestampoffset,
                            mutator=mutator)

    def savefig(self, outfile=None, categories=None, figure=1, nocaption=False, fontsize=10, boxalpha=0.15, d_category2color={'person':'green', 'vehicle':'blue', 'object':'red'}, captionoffset=(3,-18), dpi=200, textfacecolor='white', shortlabel=None, nocaption_withstring=[], timestamp=None, mutator=None, timestampoffset=(0,0), theme='dark'):
        """Save `vipy.image.Scene.show output to given file or return buffer without popping up a window"""
        fignum = figure if figure is not None else 1        
        self.show(categories=categories, figure=fignum, nocaption=nocaption, fontsize=fontsize, boxalpha=boxalpha, 
                  d_category2color=d_category2color, captionoffset=captionoffset, nowindow=True, 
                  shortlabel=shortlabel, nocaption_withstring=nocaption_withstring, timestamp=timestamp,
                  mutator=mutator, timestampoffset=timestampoffset, theme=theme)
        
        if outfile is None:
            buf = io.BytesIO()
            (W,H) = plt.figure(num=fignum).canvas.get_width_height()  # fast(ish)
            plt.figure(num=fignum).canvas.print_raw(buf)  # fast(ish), FIXME: there is a bug here with captions showing behind bboxes on macos
            img = np.frombuffer(buf.getbuffer(), dtype=np.uint8).reshape((H, W, 4))
            if figure is None:
                vipy.show.close(plt.gcf().number)   # memory cleanup (useful for video annotation on last frame)
            return vipy.image.Image(array=img, colorspace='rgba').rgb()
        else:
            vipy.show.savefig(os.path.abspath(os.path.expanduser(outfile)), figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
            return outfile

        
    
class ImageDetection(Scene):
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
                 xcentroid=None, ycentroid=None, category=None, xywh=None, ulbr=None, bbox=None, id=True):

        super().__init__(filename=filename,
                         url=url,
                         attributes=attributes,
                         array=array,
                         colorspace=colorspace)
        
        self.add_object(vipy.object.Detection(xmin=xmin,
                                              ymin=ymin,
                                              width=width,
                                              height=height,
                                              xmax=xmax,
                                              ymax=ymax,
                                              xcentroid=xcentroid,
                                              ycentroid=ycentroid,
                                              xywh=xywh if xywh is not None else (bbox.xywh() if isinstance(bbox, BoundingBox) else None),
                                              ulbr=ulbr,
                                              category=category,
                                              attributes=attributes,
                                              id=id))
        
    def __repr__(self):
        return str('<vipy.image.ImageDetection: %s, %s>' % (super().__repr__(), self._objectlist[0].__repr__()))
        
    def __eq__(self, other):
        """ImageDetection equality is defined as equivalent categories and boxes (not pixels)"""
        return self.boundingbox() == other.boundingbox() if isinstance(other, ImageDetection) else False

    def num_objects(self):
        return 1
    
    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        return cls(filename=d['filename'] if 'filename' in d else None,
                   url=d['url'] if 'url' in d else None,
                   category=d['category'] if 'category' in d else None,
                   attributes=d['attributes'] if 'attributes' in d else None,
                   colorspace=d['colorspace'] if 'colorspace' in d else None,
                   array=np.array(d['array'], dtype=np.uint8) if 'array' in d and d['array'] is not None else None,                                                         
                   xmin=d['xmin'] if 'xmin' in d else None,
                   ymin=d['ymin'] if 'ymin' in d else None,                   
                   xmax=d['xmax'] if 'xmax' in d else None,
                   ymax=d['ymax'] if 'ymax' in d else None,
                   id=d['id'] if 'id' in d else None)

    def boundingbox(self):
        return vipy.geometry.BoundingBox(ulbr=self._objectlist[0].ulbr())

    def crop(self):
        """Crop the image using the bounding box and return a `vipy.image.Image` for the cropped pixels"""
        return vipy.image.Image.cast(self.clone())._crop(self.boundingbox())

    
    
def mutator_show_trackid(n_digits_in_trackid=None):
    """Mutate the image to show track ID with a fixed number of digits appended to the category as (####)"""
    return lambda im, k=None: (im.objectmap(lambda o: o.new_category('%s (%s)' % (o.category(), o.attributes['__trackid'][0:n_digits_in_trackid]))
                                            if o.has_attribute('__trackid') else o))

def mutator_show_trackindex():
    """Mutate the image to show track index appended to the category as (####)"""
    return lambda im, k=None: (im.objectmap(lambda o: o.new_category('%s (%d)' % (o.category(), int(o.attributes['__track_index']))) if o.has_attribute('__track_index') else o))

def mutator_show_track_only():
    """Mutate the image to show track as a consistently colored box with no categories"""
    f = mutator_show_trackindex()
    return lambda im, k=None, f=f: f(im).objectmap(lambda o: o.new_category('__%s' % o.category()))  # prepending __ will not show it, but will color boxes correctly
    
def mutator_show_noun_only(nocaption=False):
    """Mutate the image to show the noun only.  
    
    Args:
        nocaption: [bool] If true, then do not display the caption, only consistently colored boxes for the noun. 
    
    ..note:: To color boxes by track rather than noun, use `vipy.image.mutator_show_trackonly`
    """
    return lambda im, k=None: (im.objectmap(lambda o: o.new_category('\n'.join([('__'+n if nocaption else n) for n in o.attributes['__track_category']])) if o.has_attribute('__track_category') else o))

def mutator_show_verb_only():
    """Mutate the image to show the verb only"""
    return lambda im, k=None: (im.objectmap(lambda o: o.new_category('\n'.join([v for v in o.attributes['__activity_category']])) if o.has_attribute('__activity_category') else o))

def mutator_show_noun_or_verb():
    """Mutate the image to show the verb only if it is non-zero else noun"""
    return lambda im,k=None: (im.objectmap(lambda o: o.new_category('\n'.join([v if len(v)>0 else n for (n,v) in zip(o.attributes['__track_category'], o.attributes['__activity_category'])])) if o.has_attribute('__track_category') and o.has_attribute('__activity_category') else o))

def mutator_show_noun_verb():
    """Mutate the image to show the category as 'Noun Verb1\nNoun Verb2'"""
    return lambda im, k=None: (im.objectmap(lambda o: o.new_category('\n'.join(['%s %s' % (n.capitalize().replace('_',' '),
                                                                                       (v.replace('%s_'%n.lower(),'',1) if v.lower().startswith(n.lower()) else v).replace('_',' '))
                                                                            for (n,v) in zip(o.attributes['__track_category'], o.attributes['__activity_category'])]))
                                            if o.has_attribute('__track_category') and o.has_attribute('__activity_category') else o))
    
def mutator_show_trackindex_verbonly(confidence=True, significant_digits=2):
    """Mutate the image to show boxes colored by track index, and only show 'verb' captions with activity confidence, sorted in decreasing order"""
    f = mutator_show_trackindex()
    return lambda im, k=None, f=f: f(im).objectmap(lambda o: o.new_category('__%s' % o.category()) if (len(o.attributes['__track_category']) == 1 and len(o.attributes['__activity_category'][0]) == 0) else o.new_category('\n'.join(['%s %s' % (v, ('(%1.2f)'%float(c)) if (confidence is True and c is not None) else '') for (n,v,c) in sorted(zip(o.attributes['__track_category'], o.attributes['__activity_category'], o.attributes['__activity_conf']), key=lambda x: float(x[2]) if x[2] else 0, reverse=True)])))


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

def RandomScene(rows=None, cols=None, num_detections=8, num_keypoints=8, num_tags=4, num_objects=None, url=None):
    """Return a uniform random color `vipy.image.Scene` of size (rows, cols) with a specified number of vipy.object.Object` objects"""    
    im = Scene(array=RandomImage(rows, cols).array()) if url is None else Scene(url=url)
    (rows, cols) = im.shape()
    objects = []
    if num_objects:
        (num_detection, num_keypoints) = (num_objects//2, num_objects//2)
    if num_detections:
        objects += [vipy.object.Detection('obj%04d' % k, xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16), width=np.random.randint(16,cols), height=np.random.randint(16,rows), confidence=float(np.random.rand())) for k in range(num_detections)]
    if num_keypoints:
        objects += [vipy.object.Keypoint2d(category='kp%02d' % (k%20), x=np.random.randint(0,cols - 16), y=np.random.randint(0,rows - 16), radius=np.random.randint(0.01*cols, 0.1*cols), confidence=float(np.random.rand())) for k in range(num_keypoints)]
    if num_tags:
        im.add_soft_tags([('tag%d'%k, float(np.random.rand())) for k in range(num_tags)])
            
    return im.objects(objects)
    

def owl():
    """Return a superb owl image for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/512px-Bubo_virginianus_06.jpg',
                 filename=vipy.util.tocache('owl.jpg'), # to avoid redownload 
                 objects=[vipy.object.Detection('Great Horned Owl', xmin=93, ymin=85, width=373, height=560)]).mindim(1024)

def vehicles():
    """Return a highway scene with the four highest confidence vehicle detections for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/3/3e/I-80_Eastshore_Fwy.jpg',
                 filename=vipy.util.tocache('vehicles.jpg'), # to avoid redownload                 
                 category='Highway',
                 objects=[vipy.object.Detection(category="car", xywh=(473.0, 592.2, 92.4, 73.4)),
                          vipy.object.Detection(category="car", xywh=(1410.0, 756.1, 175.2, 147.3)),
                          vipy.object.Detection(category="car", xywh=(316.9, 640.1, 119.4, 119.5)),
                          vipy.object.Detection(category="car", xywh=(886.9, 892.9, 223.8, 196.6))]).mindim(1024)

def people():
    """Return a crowd scene with the four highest confidence person detections for testing"""
    return Scene(url='https://upload.wikimedia.org/wikipedia/commons/b/be/July_4_crowd_at_Vienna_Metro_station.jpg',
                 filename=vipy.util.tocache('people.jpg'), # to avoid redownload
                 category='crowd',
                 objects=[vipy.object.Detection(category="person", xywh=(1.8, 1178.7, 574.1, 548.0)),
                          vipy.object.Detection(category="person", xywh=(1589.4, 828.3, 363.0, 887.7)),
                          vipy.object.Detection(category="person", xywh=(1902.9, 783.1, 250.8, 825.8)),
                          vipy.object.Detection(category="person", xywh=(228.2, 948.7, 546.8, 688.5))]).mindim(1024)

    
    
class Transform():
    """Transforms are static methods that implement common transformation patterns used in distributed processing.  

       These are useful for parallel processing of noisy or corrupted images when anonymous functions are not supported (e.g. multiprocessing)
 
       See also: `vipy.dataset.Dataset.minibatch` for parallel processing of batches of images for downloading, loading, resizing, cropping, augmenting, tensor prep etc.
    """
    
    @staticmethod
    def load(im):
        try:
            return im.clone().load()
        except:
            return im.flush()

    @staticmethod
    def download(im):
        try:
            return im.clone().download()
        except:
            return im.flush()

    @staticmethod
    def is_loaded(im):
        return im.is_loaded()

    @staticmethod
    def mindim(im, mindim=256):
        try:
            return im.clone().load().mindim(mindim)
        except:
            return im.flush()

        
    @staticmethod
    def thumbnail(im, mindim=64, outfile=None):
        try:
            return im.clone().load().mindim(mindim).save(outfile if outfile else tocache(shortuuid(8)+'.jpg'))
        except:
            return im.flush()

    @staticmethod
    def saveas(im, filename):
        try:
            return im.clone().load().saveas(filename)
        except:
            return im.flush()
        
    @staticmethod
    def annotate(im, mindim=64, outfile=None):
        try:
            return im.clone().load().mindim(mindim).annotate().save(outfile if outfile else tocache(shortuuid(8)+'.jpg'))
        except:
            return im.flush()
        
    @staticmethod
    def centersquare_32x32_normalized(im):
        return im.clone().load().rgb().centersquare().resize(32,32).gain(1/255) if not im.loaded() else im

    @staticmethod
    def centersquare_32x32_lum_normalized(im):
        return im.clone().load().centersquare().lum().resize(32,32).gain(1/255) if not im.loaded() else im
    
    @staticmethod
    def centersquare_256x256_normalized(im):
        return im.clone().load().rgb().centersquare().resize(256,256).gain(1/255) if not im.loaded() else im

    @staticmethod
    def mindim256_normalized(im):
        return im.clone().load().rgb().mindim(256).gain(1/255) if not im.loaded() else im
    
    @staticmethod
    def tensor(image, shape=None, gain=None, mindim=None, colorspace=None, centersquare=None, tensor=None, ignore_errors=False, jitter=None, num_augmentations=None):
        try:
            im = image.clone()
            if colorspace == 'lum':
                im = im.lum()
            if colorspace == 'rgb':
                im = im.rgb()
            if colorspace == 'float':
                im = im.float()
            if jitter == 'randomcrop':
                import vipy.noise                  
                im = vipy.noise.randomcrop(im)                
            if centersquare:
                im = im.centersquare()
            if shape is not None:
                im = im.resize(*shape)
            if mindim:
                im = im.mindim(mindim)
            if gain is not None:
                im = im.gain(gain)
            if tensor:
                im = im.torch()  # CHW
            if num_augmentations:
                augmentations = np.stack([np.atleast_3d(Transform.tensor(image, shape=shape, gain=gain, mindim=mindim, colorspace=colorspace, centersquare=centersquare, ignore_errors=ignore_errors, jitter=jitter).array())
                                          for k in range(num_augmentations+1)], axis=3)  # +1 for mean 
                return image.clone().array(augmentations)  # packed nd-array, use im.torch('NCHW') to access
                
            return im
        
        except KeyboardInterrupt:
            raise
        except:
            if not ignore_errors:
                raise
            return None

    @staticmethod
    def to_tensor(**kwargs):
        return functools.partial(Transform.tensor, **kwargs)

    @staticmethod
    def is_transformed(im):
        return im is not None
    
