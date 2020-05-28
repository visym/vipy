import os
import PIL
import PIL.Image
import platform
import dill
import vipy.show
from vipy.show import imshow, imbbox, savefig, colorlist
from vipy.util import isnumpy, isurl, isimageurl, \
    fileext, tempimage, mat2gray, imwrite, imwritegray, \
    tempjpg, filetail, isimagefile, remkdir, hasextension, \
    try_import, tolist, islistoflists, istupleoftuples, isstring, \
    istuple, islist, isnumber, isnumpyarray
from vipy.geometry import BoundingBox, imagebox
import vipy.object
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
import warnings
import base64
import types
import hashlib
from itertools import repeat
import atexit


class Image(object):
    """vipy.image.Image class
    
    The vipy image class provides a fluent, lazy interface for representing, transforming and visualizing images.
    The following constructors are supported:

    >>> im = vipy.image.Image(filename="/path/to/image.ext")
    
        All image file formats that are readable by PIL are supported here.

    >>> im = vipy.image.Image(url="http://domain.com/path/to/image.ext")
    
        The image will be downloaded from the provided url and saved to a temporary filename.
        The environment variable VIPY_CACHE controls the location of the directory used for saving images, otherwise this will be saved to the system temp directory.

    >>> im = vipy.image.Image(url="http://domain.com/path/to/image.ext", filename="/path/to/new/image.ext")

        The image will be downloaded from the provided url and saved to the provided filename.
        The url() method provides optional basic authentication set for username and password

    >>> im = vipy.image.Image(array=img, colorspace='rgb')

        The image will be constructed from a provided numpy array 'img', with an associated colorspace.  The numpy array and colorspace can be one of the following combinations:

            'rgb': uint8, three channel (red, green, blue)
            'rgba':  uint8, four channel (rgb + alpha)
            'bgr': uint8, three channel (blue, green, red), such as is returned from cv2.imread()
            'bgra':  uint8, four channel
            'hsv':  uint8, three channel (hue, saturation, value)
            'lum;:  uint8, one channel, luminance (8 bit grey level)
            'grey':  float32, one channel in range [0,1] (32 bit intensity)
            'float':  float32, any channel in range [-inf, +inf]

        The most general colorspace is 'float' which is used to manipulate images prior to network encoding, such as applying bias. 
    """

    def __init__(self, filename=None, url=None, array=None, colorspace=None, attributes=None):
        # Private attributes
        self._ignoreErrors = False  # ignore errors during fetch (broken links)
        self._urluser = None      # basic authentication set with url() method
        self._urlpassword = None  # basic authentication set with url() method
        self._urlsha1 = None      # file hash if known
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
        self.array(array)

        # Guess RGB colorspace if three channel uint8 if colorspace is not provided
        colorspace = 'rgb' if (self.isloaded() and self.channels() == 3 and self._array.dtype == np.uint8 and colorspace is None) else colorspace
        self.colorspace(colorspace)
        
        # Public attributes: passed in as a dictionary
        self.attributes = {} 
        if attributes is not None:
            assert isinstance(attributes, dict), "Attributes must be dictionary"
            self.attributes = attributes

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
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        return str('<vipy.image: %s>' % (', '.join(strlist)))

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes of the Image() object"""
        d = {'filename':self.filename(),
             'url':self.url(),
             'attributes':self.attributes,
             'height':None,
             'width':None,
             'channels':None,
             'colorspace':None,
             'array':None}
        if self.isloaded():
            d['height'] = self.height()
            d['width'] = self.width()
            d['channels'] = self.channels()
            d['colorspace'] = self.colorspace()
            d['array'] = self.array()
        return {'image':d}

    def loader(self, f):
        """Lambda function to load an unsupported image filename to a numpy array"""
        self._loader = f
        return self

    def load(self, ignoreErrors=False, verbose=False):
        """Load image to cached private '_array' attribute and return Image object"""
        try:
            # Return if previously loaded image
            if self._array is not None:
                return self

            # Download URL to filename
            if self._url is not None:
                self.download(ignoreErrors=ignoreErrors, verbose=verbose)

            # Load filename to numpy array
            if self._loader is not None:
                self._array = self._loader(self._filename).astype(np.float32)  # forcing float32
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
                    warnings.warn('unknown colorspace for image "%s" - attempting to coerce to colorspace=float' % str(self._filename))
                    self._array = np.float32(self._array)
                    self.colorspace('float')                    
            elif hasextension(self._filename):
                raise ValueError('Non-standard image extensions require a custom loader')
            else:
                # Attempting to open it anyway, may be an image file without an extension. Cross your fingers ...
                self._array = np.array(PIL.Image.open(self._filename))  # RGB order!

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                if verbose is True:
                    warnings.warn('[vipy.image][WARNING]: IO error "%s" -> "%s" - Ignoring. ' % (self.url(), self.filename()))
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                if verbose is True:
                    warnings.warn('[vipy.image][WARNING]: Load error for image "%s" - Ignoring' % self.filename())
                self._array = None
            else:
                raise

        return self

    def download(self, ignoreErrors=False, timeout=10, verbose=False):
        """Download URL to filename provided by constructor, or to temp filename"""
        if self._url is None and self._filename is not None:
            return self
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[vipy.image.download][ERROR]: '
                             'Invalid URL "%s" ' % self._url)

        if self._filename is None:
            if 'VIPY_CACHE' in os.environ:
                self._filename = os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(self._url))
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
                                         sha1=self._urlsha1,
                                         username=self._urluser,
                                         password=self._urlpassword)
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
            if self._ignoreErrors or ignoreErrors:
                if verbose is True:
                    warnings.warn('[vipy.image][WARNING]: download failed - Ignoring image')
                self._array = None
            else:
                raise

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                if verbose:
                    warnings.warn('[vipy.image][WARNING]: IO error downloading "%s" -> "%s" - Ignoring' % (self.url(), self.filename()))
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                if verbose:
                    warnings.warn('[vipy.image][WARNING]: load error for image "%s"' % self.filename())
            else:
                raise

        return self

    def reload(self):
        """Flush the image buffer to force reloading from file or URL"""
        return self.clone(flush=True).load()

    def isloaded(self):
        return self._array is not None

    def channels(self):
        """Return integer number of color channels"""
        return 1 if self.load().array().ndim == 2 else self.load().array().shape[2]

    def iscolor(self):
        """Color images are three channel or four channel with transparency, float32 or uint8"""
        return self.channels() == 3 or self.channels() == 4

    def istransparent(self):
        """Color images are three channel or four channel with transparency, float32 or uint8"""
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
        return self.load().array().shape[1]

    def height(self):
        return self.load().array().shape[0]

    def shape(self):
        """Return the (height, width) or equivalently (rows, cols) of the image"""
        return (self.load().height(), self.width())

    def centroid(self):
        """Return the real valued center pixel coordinates of the image (col=x,row=y)"""
        return (self.load().width() / 2.0, self.height() / 2.0)

    def centerpixel(self):
        """Return the integer valued center pixel coordinates of the image (col=i,row=j)"""
        c = np.round(self.centroid())
        return (int(c[0]), int(c[1]))
    
    def array(self, np_array=None, copy=False):
        """Replace self._array with provided numpy array"""
        if np_array is None:
            return self._array
        elif isnumpyarray(np_array):
            assert np_array.dtype == np.float32 or np_array.dtype == np.uint8, "Invalid input - array() must be type uint8 or float32 and not type='%s'" % (str(np_array.dtype))
            self._array = np.copy(np_array) if copy else np_array  # reference or copy
            self.colorspace(None)  # must be set with colorspace() after array() but before _convert()
            return self
        else:
            raise ValueError('Invalid input - array() must be numpy array and not "%s"' % (str(type(np_array))))

    def channel(self, k):
        """Return a cloned Image() object for the kth channel"""
        assert k < self.channels(), "Requested channel=%d must be within valid channels=%d" % (k, self.channels())
        assert self.channels() > 1, "Channel() only valid for multi-channel image"
        im = self.clone().load()
        im._array = im._array[:,:,k]
        im._colorspace = 'lum'
        return im

    def red(self):
        """Return red channel as a cloned Image() object"""
        assert self.channels() >= 3, "Must be color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(0)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(3)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())

    def green(self):
        """Return green channel as a cloned Image() object"""
        assert self.channels() >= 3, "Must be color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(1)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(1)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())

    def blue(self):
        """Return blue channel as a cloned Image() object"""
        assert self.channels() >= 3, "Must be color image"
        if self.colorspace() in ['rgb', 'rgba']:
            return self.channel(2)
        elif self.colorspace() in ['bgr', 'bgra']:
            return self.channel(0)
        else:
            raise ValueError('Invalid colorspace "%s" does not contain red channel' % self.colorspace())                

    def alpha(self):
        """Return alpha (transparency) channel as a cloned Image() object"""
        assert self.channels() == 4 and self.colorspace() in ['rgba', 'bgra'], "Must be four channnel color image"
        return self.channel(3)
        
    def fromarray(self, data):
        """Alias for array(data, copy=True), set new array() with a numpy array copy"""
        return self.array(data, copy=True)
    
    def tonumpy(self):
        """Alias for numpy()"""
        return self.numpy()

    def numpy(self):
        """Convert vipy.image.Image to numpy array, returns writeable array"""
        self.load()
        self._array = np.copy(self._array) if not self._array.flags['WRITEABLE'] else self._array  # triggers copy         
        return self._array

    def zeros(self):
        self._array = 0*self.load()._array
        return self

    def pil(self):
        """Convert vipy.image.Image to PIL Image, by reference"""
        if self.colorspace() == 'float':
            warnings.warn('Coercing generic colorspace=float to colorspace=rgb for image transformation')
            self.rgb()
        return PIL.Image.fromarray(self.tonumpy())

    def torch(self, order='CHW'):
        """Convert the batch of 1 HxWxC images to a 1xCxHxW torch tensor, by reference"""
        try_import('torch'); import torch
        img = self.numpy() if self.iscolor() else np.expand_dims(self.numpy(), 2)  # HxW -> HxWx1
        img = np.expand_dims(img,0)
        img = img.transpose(0,3,1,2) if order == 'CHW' else img  # HxWxC or CxHxW
        return torch.from_numpy(img)  

    def fromtorch(self, x):
        """Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace"""
        try_import('torch'); import torch
        assert isinstance(x, torch.Tensor), "Invalid input type '%s'- must be torch.Tensor" % (str(type(x)))
        img = np.copy(np.squeeze(x.permute(2,3,1,0).detach().numpy()))  # 1xCxHxW -> HxWxC, copied
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace  # assumed
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace        
        return Image(array=img, colorspace=colorspace)
    
    def nofilename(self):
        self._filename = None
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

    def url(self, url=None, username=None, password=None, sha1=None, ignoreUrlErrors=None):
        """Image URL and URL download properties"""
        if url is not None:
            self._url = url  # this does not change anything else (e.g. the associated filename), better to use constructor 
        if username is not None:
            self._urluser = username  # basic authentication
        if password is not None:
            self._urlpassword = password  # basic authentication
        if sha1 is not None:
            self._urlsha1 = sha1  # file integrity
        if ignoreUrlErrors is not None:
            self._ignoreErrors = ignoreUrlErrors
        if url is None and username is None and password is None and sha1 is None and ignoreUrlErrors is None:
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
    
    def getattribute(self, key):
        if self.attributes is not None and key in list(self.attributes.keys()):
            return self.attributes[key]
        else:
            return None

    def hasattribute(self, key):
        return self.attributes is not None and key in self.attributes

    def hasurl(self):
        return self._url is not None and isurl(self._url)

    def hasfilename(self):
        return self._filename is not None and os.path.exists(self._filename)

    def clone(self, flushforward=False, flushbackward=False, flush=False):
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
        else:
            im = copy.deepcopy(self)            
        return im
    

    def _interp_string_to_pil_interpolation(self, interp):
        """Internal function to convert interp string to interp object"""
        assert interp in ['bilinear', 'bicubic', 'nearest'], "Invalid interp - Must be in ['bilinear', 'bicubic', 'nearest']"
        if interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'nearest':
            return PIL.Image.NEAREST
        else:
            raise  # should never get here
        
    # Spatial transformations
    def resize(self, cols=None, rows=None, width=None, height=None, interp='bilinear'):
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

        else:
            self._array = np.array(self.load().pil().resize((cols, rows), self._interp_string_to_pil_interpolation(interp)))
        return self

    def resize_like(self, im, interp='bilinear'):
        """Resize image buffer to be the same size as the provided vipy.image.Image()"""
        assert isinstance(im, Image), "Invalid input - Must be vipy.image.Image()"
        return self.resize(im.width(), im.height(), interp=interp)
    
    def rescale(self, scale=1, interp='bilinear'):
        """Scale the image buffer by the given factor - NOT idemponent"""
        (height, width) = self.load().shape()
        self._array = np.array(self.pil().resize((int(np.round(scale * width)), int(np.round(scale * height))), self._interp_string_to_pil_interpolation(interp)))
        return self

    def maxdim(self, dim, interp='bilinear'):
        """Resize image preserving aspect ratio so that maximum dimension of image = dim"""
        return self.rescale(float(dim) / float(np.maximum(self.height(), self.width())), interp=interp)

    def mindim(self, dim, interp='bilinear'):
        """Resize image preserving aspect ratio so that minimum dimension of image = dim"""
        return self.rescale(float(dim) / float(np.minimum(self.height(), self.width())), interp=interp)

    def _pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode, dx=padwidth, dy=padheight"""
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
        self._array = np.pad(self.load().array(),
                             pad_width=pad_shape,
                             mode='constant',
                             constant_values=0)
        return self

    def zeropadlike(self, width, height):
        """Zero pad the image balancing the border so that the resulting image size is (width, height)"""
        assert width >= self.width() and height >= self.height(), "Invalid input - final (width=%d, height=%d) must be greater than current image size (width=%d, height=%d)" % (width, height, self.width(), self.height())
        return self.zeropad( (int(np.floor((width - self.width())/2)), int(np.ceil((width - self.width())/2))),
                             (int(np.floor((height - self.height())/2)), int(np.ceil((height - self.height())/2))))
                            
    def meanpad(self, padwidth, padheight):
        """Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding,, and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding"""        
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth, padwidth)
        if not isinstance(padheight, tuple):
            padheight = (padheight, padheight)
        if self.iscolor():
            pad_shape = (padheight, padwidth, (0, 0))
        else:
            pad_shape = (padheight, padwidth)

        assert all([x>=0 for x in padheight]) and all([x>=0 for x in padwidth]), "padding must be positive"
        mu = self.meanchannel()  
        self._array = np.pad(self.load().array(),
                             pad_width=pad_shape,
                             mode='constant',
                             constant_values=mu)
        return self

    def minsquare(self):
        """Crop image of size (HxW) to (min(H,W), min(H,W)), keeping upper left corner constant"""
        S = np.min(self.load().shape())
        return self._crop(BoundingBox(xmin=0, ymin=0, width=S, height=S))

    def maxsquare(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding, keeping upper left corner constant"""
        S = np.max(self.load().shape())
        dW = S - self.width()
        dH = S - self.height()
        return self.zeropad((0,dW), (0,dH))._crop(BoundingBox(0, 0, width=S, height=S))

    def maxmatte(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte"""
        S = np.max(self.load().shape())
        dW = S - self.width()
        dH = S - self.height()
        return self.zeropad((int(np.floor(dW//2)), int(np.ceil(dW//2))), (int(np.floor(dH//2)), int(np.ceil(dH//2))))._crop(BoundingBox(0, 0, width=S, height=S))
    
    def centersquare(self):
        """Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant"""
        N = int(np.min(self.shape()))
        return self._crop(BoundingBox(xcentroid=self.width() / 2.0, ycentroid=self.height() / 2.0, width=N, height=N))

    def centercrop(self, height, width):
        """Crop image of size (height x width) in the center, keeping the image centroid constant"""
        return self._crop(BoundingBox(xcentroid=self.width() / 2.0, ycentroid=self.height() / 2.0, width=width, height=height))

    def _crop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, clipping the box to the image rectangle"""
        assert isinstance(bbox, BoundingBox) and bbox.valid(), "Invalid vipy.geometry.BoundingBox() input"""
        if not bbox.isdegenerate() and bbox.hasoverlap(self.load().array()):
            bbox = bbox.imclip(self.load().array()).int()
            self._array = self.array()[bbox.ymin():bbox.ymax(),
                                       bbox.xmin():bbox.xmax()]
        else:
            warnings.warn('BoundingBox for crop() does not intersect image rectangle - Ignoring')
        return self
    
    def fliplr(self):
        """Mirror the image buffer about the vertical axis - Not idemponent"""
        self._array = np.fliplr(self.load().array())
        return self

    def imagebox(self):
        """Return the bounding box for the image rectangle"""
        return BoundingBox(xmin=0, ymin=0, width=self.width(), height=self.height())
    
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
                self._array = np.dstack((img, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))
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
                warnings.warn('Float image will be rescaled with self.mat2gray() into the range float32 [0,1]')
                img = self.mat2gray().array()
            if not self.channels() in [1,3]:
                raise ValueError('Float image must be single channel or three channel RGB in the range float32 [0,1] prior to conversion')
            if self.channels() == 3:  # assumed RGB
                self._array = np.uint8(255 * self.array())   # float32 RGB [0,1] -> uint8 RGB [0,255]
                self.colorspace('rgb')
            else:
                self._array = (1.0 / 255.0) * np.array(PIL.Image.fromarray(np.uint8(255 * self.array())).convert('L')).astype(np.float32)  # float32 RGB [0,1] -> float32 gray [0,1]                
                self.colorspace('grey')
            self._convert(to)
        elif self.colorspace() is None:
            raise ValueError('Colorspace must be initialized by constructor or colorspace() to allow for colorspace conversion')
        else:
            raise ValueError('unsupported colorspace "%s"' % self.colorspace())

        self.colorspace(to)
        return self

    def rgb(self):
        """Convert the image buffer to three channel RGB uint8 colorspace"""
        return self._convert('rgb')

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
        """Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float' """
        self.array((self.load().float().array()) - float(self.min()) / float(self.max() - self.min()))
        return self.colorspace('float')

    def mat2gray(self, min=None, max=None):
        """Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace"""
        self.array(mat2gray(np.float32(self.load().float().array()), min, max))
        return self.colorspace('float')
        return self

    def gain(self, g):
        """Elementwise multiply gain to image array, Gain should be broadcastable to array().  This forces the colospace to 'float'"""
        self.array(np.multiply(self.load().float().array(), g))
        return self.colorspace('float')

    def bias(self, b):
        """Add a bias to the image array.  Bias should be broadcastable to array().  This forces the colorspace to 'float'"""
        self.array(self.load().float().array() + b)
        return self.colorspace('float')
    
    # Image statistics
    def stats(self):
        print(self)
        print('  Channels: %d' % self.channels())
        print('  Shape: %s' % str(self.shape()))
        print('  min: %d' % self.min())
        print('  max: %d' % self.max())
        print('  mean: %s' % str(self.mean()))
        print('  channel mean: %s' % str(self.meanchannel()))        
    
    def min(self):
        return np.min(self.load().array().flatten())

    def max(self):
        return np.max(self.load().array().flatten())

    def mean(self):
        """Mean over all pixels"""
        return np.mean(self.load().array().flatten())

    def meanchannel(self):
        """Mean per channel over all pixels"""
        return np.mean(self.load().array(), axis=(0, 1)).flatten()
    
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
    
    def show(self, figure=None, nowindow=False):
        """Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object"""
        assert self.load().isloaded(), 'Image not loaded'
        imshow(self.clone().rgb().numpy(), fignum=figure, nowindow=nowindow)
        return self

    def save(self, filename):
        """Save the current image to a new filename and return the image object"""
        assert filename is not None, "Invalid filename - must be path to new image filename"
        return self.filename(self.saveas(filename))
        
        
    # Image export
    def saveas(self, filename, writeas=None):
        """Save current buffer (not including drawing overlays) to new filename and return filename"""
        assert filename is not None, 'Valid filename="/path/to/image.ext" must be provided'
        if self.colorspace() in ['gray']:
            imwritegray(self.grayscale()._array, filename)
        elif self.colorspace() != 'float':
            imwrite(self.load().array(), filename, writeas=writeas)
        else:
            raise ValueError('Convert float image to RGB or gray first. Try self.mat2gray()')
        return filename

    def saveastmp(self):
        """Save current buffer to temp JPEG filename and return filename.  Alias for savetmp()"""
        return self.saveas(tempjpg())

    def savetmp(self):
        """Save current buffer to temp JPEG filename and return filename.   Alias for saveastmp()"""
        return self.saveastmp()

    def base64(self):
        """Export a base64 encoding of the image suitable for embedding in an html page"""
        buf = io.BytesIO()
        self.clone().rgb().pil().save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue())
        
    def html(self, alt=None):
        """Export a base64 encoding of the image suitable for embedding in an html page, enclosed in <img> tag
           
           Returns:
              -string:  <img src="data:image/jpeg;charset=utf-8;base64,%s" alt="%s"> containing base64 encoded JPEG and alt text        
        """
        b = self.base64().decode('ascii')
        alt_text = alt if alt is not None else self.filename()
        return '<img src="data:image/jpeg;charset=utf-8;base64,%s" alt="%s">' % (b, alt_text)

    def savefig(self, filename=None):
        """Save last figure output from self.show() with drawing overlays to provided filename and return filename"""
        self.show(figure=1, nowindow=True)  # sets figure dimensions, does not display window
        if filename is None:
            (W,H) = plt.figure(1).canvas.get_width_height()  # fast
            buf = io.BytesIO()
            plt.figure(1).canvas.print_raw(buf)  # fast
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((H, W, 4))  # RGBA
            plt.close(1)  # memory cleanup
            return vipy.image.Image(array=img, colorspace='rgba')
        else:
            return savefig(filename)

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

    
class ImageCategory(Image):
    """vipy ImageCategory class

    This class provides a representation of a vipy.image.Image with a category. 

    Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label')

    >>> im = vipy.image.ImageCategory(filename='/path/to/dog_image.ext', category='dog')
    >>> im = vipy.image.ImageCategory(url='http://path/to/dog_image.ext', category='dog')
    >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog')

    """
    
    def __init__(self, filename=None, url=None, category=None, label=None,
                 attributes=None, array=None, colorspace=None):
        # Image class inheritance
        super(ImageCategory, self).__init__(filename=filename,
                                            url=url,
                                            attributes=attributes,
                                            array=array,
                                            colorspace=colorspace)
        assert not (category is not None and label is not None), "Define either category or label kwarg, not both"
        self._category = category if category is not None else label
        
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        return str('<vipy.imagecategory: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        return self._category.lower() == other._category.lower() if isinstance(other, ImageCategory) else False

    def __ne__(self, other):
        return self._category.lower() != other._category.lower()

    def is_(self, other):
        return self.__eq__(other)

    def is_not(self, other):
        return self.__ne__(other)

    def dict(self):
        d = super(ImageCategory, self).dict()
        d['category'] = self.category()
        return d

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

    def vimage(self):
        """Downgrade vipy.image.ImageCategory() to vipy.image.Image() object, array() is shared"""
        return Image(filename=self.filename(), url=self.url(), attributes=self.attributes, array=self.array(), colorspace=self.colorspace())

    def image(self):
        """Alias for vimage"""
        return self.vimage()
    
    
class ImageDetection(ImageCategory):
    """vipy.image.ImageDetection class

    This class provides a representation of a vipy.image.Image with an object detection with a category and a vipy.geometry.BoundingBox

    Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label'), and BoundingBox coordinates

    >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, width=100, height=100)
    >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, xmax=100, ymax=100)
    >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xcentroid=50, ycentroid=50, width=100, height=100)
    >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', bbox=vipy.geometry.BoundingBox(xmin=0, ymin=0, width=100, height=100))
    >>> im = vipy.image.ImageCategory(url='http://path/to/dog_image.ext', category='dog').boundingbox(xmin=0, ymin=0, width=100, height=100)
    >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog',  xmin=0, ymin=0, width=100, height=100)

    """
    
    def __init__(self, filename=None, url=None, category=None, attributes=None,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 width=None, bbwidth=None, height=None, bbheight=None,
                 bbox=None, array=None, colorspace=None,
                 xcentroid=None, ycentroid=None):

        # ImageCategory class inheritance
        super(ImageDetection, self).__init__(filename=filename,
                                             url=url,
                                             attributes=attributes,
                                             category=category,
                                             array=array,
                                             colorspace=colorspace)

        # Construction options
        (width, height) = (bbwidth if bbwidth is not None else width, bbheight if bbheight is not None else height)  # alias
        if bbox is not None:
            assert isinstance(bbox, BoundingBox), "Invalid bounding box"
            self.bbox = bbox
        elif xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        elif xmin is not None and ymin is not None and width is not None and height is not None:
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin, width=width, height=height)
        elif xcentroid is not None and ycentroid is not None and width is not None and height is not None:
            self.bbox = BoundingBox(xcentroid=xcentroid, ycentroid=ycentroid, width=width, height=height)
        elif (xmin is None and xmax is None and ymin is None and ymax is None and
              width is None and bbwidth is None and height is None and bbheight is None and
              bbox is None and xcentroid is None and ycentroid is None):
            # Empty box to be updated with boundingbox() method
            self.bbox = BoundingBox(xmin=0, ymin=0, width=0, height=0)
        else:
            raise ValueError('Incomplete constructor')

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url="%s"' % self.url())
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.bbox.isvalid():
            strlist.append('bbox=(xmin=%1.1f, ymin=%1.1f, width=%1.1f, height=%1.1f)' %
                           (self.bbox.xmin(), self.bbox.ymin(),self.bbox.width(), self.bbox.height()))
        return str('<vipy.imagedetection: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        """ImageDetection equality is defined as equivalent categories and boxes (not pixels)"""
        return self._category.lower() == other._category.lower() and self.bbox == other.bbox if isinstance(other, ImageDetection) else False

    def dict(self):
        d = super(ImageCategory, self).dict()
        d['boundingbox'] = self.bbox.dict()
        return d

    def detection(self):
        """Downgrade to vipy.object.Detection() object"""
        return vipy.object.Detection(category=self.category(), xywh=self.boundingbox().xywh(), attributes=self.attributes)

    def boundingbox(self, xmin=None, xmax=None, ymin=None, ymax=None,
                    bbox=None, width=None, height=None, dilate=None,
                    xcentroid=None, ycentroid=None):
        """Modify the bounding box using the provided parameters, or return the box if no parameters provided"""
        if (xmin is None and xmax is None and ymin is None and ymax is None
            and bbox is None and width is None and height is None
                and dilate is None and xcentroid is None and ycentroid is None):
            return self.bbox
        elif (xmin is not None and xmax is not None
              and ymin is not None and ymax is not None):
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin,
                                    xmax=xmax, ymax=ymax)
        elif bbox is not None:
            self.bbox = bbox
        elif (xmin is not None and ymin is not None
              and width is not None and height is not None):
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin, width=width, height=height)
        elif (xcentroid is not None and ycentroid is not None
              and width is not None and height is not None):
            self.bbox = BoundingBox(xcentroid=xcentroid,
                                    ycentroid=ycentroid,
                                    width=width,
                                    height=height)
        elif (dilate is None):
            raise ValueError('Invalid bounding box')

        if dilate is not None:
            self.bbox.dilate(dilate)

        return self

    def invalid(self):
        """An ImageDetection is invalid when the box is invalid"""
        return self.bbox.invalid()

    def isinterior(self, W=None, H=None):
        """Is the bounding box fully within the image rectangle?  Use provided image width and height (W,H) to avoid lots of reloads in some conditions"""
        (W, H) = (W if W is not None else self.width(),
                  H if H is not None else self.height())
        return (self.bbox.xmin() >= 0 and self.bbox.ymin() >= 0
                and self.bbox.xmax() <= W and self.bbox.ymax() <= H)
    
    # Spatial transformations
    def imclip(self):
        """Clip bounding box to the image rectangle, and ignore bbox if outside image rectangle.  Requires image load"""
        if self.bbox.hasintersection(vipy.geometry.imagebox(self.shape())):
            self.bbox.imclip(self.load().numpy())
        else:
            warnings.warn("Degenerate bounding box does not intersect image - Ignoring")            
        return self

    def rescale(self, scale=1, interp='bilinear'):
        """Rescale image buffer and bounding box"""
        self = super(ImageDetection, self).rescale(scale, interp=interp)
        self.bbox = self.bbox.rescale(scale)
        return self

    def resize(self, cols=None, rows=None, interp='bilinear'):
        """Resize image buffer and bounding box so that the image buffer is size (height=cols, width=row).  If only cols or rows is provided, then scale the image appropriately"""
        assert cols is not None or rows is not None, "Invalid input"
        sx = (float(cols) / self.width()) if cols is not None else None
        sy = (float(rows) / self.height()) if rows is not None else None
        sx = sy if sx is None else sx
        sy = sx if sy is None else sy
        self.bbox.scalex(sx)
        self.bbox.scaley(sy)
        if sx == sy:
            self = super(ImageDetection, self).rescale(sx, interp=interp)  # Warning: method resolution order for multi-inheritance
        else:
            self = super(ImageDetection, self).resize(cols, rows, interp=interp)
        return self

    
    def fliplr(self):
        """Mirror buffer and bounding box around vertical axis"""
        self.bbox.fliplr(width=self.width())
        self = super(ImageDetection, self).fliplr()
        return self

    def crop(self):
        """Crop image using stored bounding box, then set the bounding box equal to the new image rectangle"""
        self = super(ImageDetection, self)._crop(self.bbox)
        self.bbox = BoundingBox(xmin=0, ymin=0, xmax=self.width(), ymax=self.height())
        return self

    def mindim(self, dim, interp='bilinear'):
        """Resize image preserving aspect ratio so that minimum dimension of image = dim"""
        return super(ImageDetection, self).mindim(dim, interp=interp)  # calls self.rescale() which will update boxes

    def maxdim(self, dim, interp='bilinear'):
        """Resize image preserving aspect ratio so that maximum dimension of image = dim"""        
        return super(ImageDetection, self).maxdim(dim, interp=interp)  # calls self.rescale() will will update boxes

    def centersquare(self):
        """Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant, new bounding box may be degenerate"""
        (H,W) = self.shape()
        self = super(ImageDetection, self).centersquare()
        (dy, dx) = ((H - self.height())/2.0, (W - self.width())/2.0)        
        self.bbox.translate(-dx, -dy)
        return self
    
    def dilate(self, s):
        """Dilate bounding box by scale factor"""
        self.bbox = self.bbox.dilate(s)
        return self

    def _pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode"""
        self = super(ImageDetection, self)._pad(dx, dy, mode)
        self.bbox = self.bbox.translate(dx if not isinstance(dx, tuple) else dx[0], dy if not isinstance(dy, tuple) else dy[0])        
        return self

    def zeropad(self, dx, dy):
        """Pad image with dx=(leftpadwidth,rightpadwidth) or dx=bothpadwidth to zeropad left and right, dy=(toppadheight,bottompadheight) or dy=bothpadheight to zeropad top and bottom"""
        self = super(ImageDetection, self).zeropad(dx, dy)
        self.bbox = self.bbox.translate(dx if not isinstance(dx, tuple) else dx[0], dy if not isinstance(dy, tuple) else dy[0])
        return self

    def meanpad(self, dx, dy):
        """Pad image using np.pad constant where constant is the RGB mean per image"""
        self = super(ImageDetection, self).meanpad(dx, dy)
        self.bbox = self.bbox.translate(dx if not isinstance(dx, tuple) else dx[0], dy if not isinstance(dy, tuple) else dy[0])        
        return self
        
    # Image export    
    def show(self, figure=None, nowindow=False):
        """Show the ImageDetection in the provided figure number"""
        if figure is not None:
            assert isinstance(figure, int) and figure > 0, "Invalid figure number, provide an integer > 0"
        self.load()
        if self.bbox.invalid():
            warnings.warn('Ignoring invalid bounding box "%s"' % str(self.bbox))
        if self.bbox.valid() and self.bbox.hasoverlap(self.array()) and self.bbox.shape() != self._array.shape[0:2]:
            self.imclip()  # crop bbox to image rectangle for valid overlay image
            imbbox(self.clone().rgb()._array, self.bbox.xmin(),
                   self.bbox.ymin(), self.bbox.xmax(), self.bbox.ymax(),
                   bboxcaption=self.category(),
                   fignum=figure, nowindow=nowindow)
        else:
            # Do not display the box if the box is degenerate or equal to the image rectangle
            super(ImageDetection, self).show(figure=figure, nowindow=nowindow)
        return self

    def rectangular_mask(self, W=None, H=None):
        """Return a binary array of the same size as the image (or using the
        provided image width and height (W,H) size to avoid an image load),
        with ones inside the bounding box, do not modify this image or this bounding box"""
        if (W is None or H is None):
            (H, W) = (int(np.round(self.height())),
                      int(np.round(self.width())))
        immask = np.zeros((H, W)).astype(np.uint8)
        if self.bbox.hasoverlap(immask):
            bb = self.bbox.clone().imclip(immask).int()
            immask[bb.ymin():bb.ymax(), bb.xmin():bb.xmax()] = 1
        return immask

    def setzero(self, bbox=None):
        """Set all image values within the bounding box (or provided bbox) to zero, triggers load()"""
        if bbox is not None:
            assert isinstance(bbox, BoundingBox), "Invalid bounding box - Must be vipy.geometry.BoundingBox() "
        bbox = self.bbox if bbox is None else bbox
        self.load().array()[int(bbox.ymin()):int(bbox.ymax()),
                            int(bbox.xmin()):int(bbox.xmax())] = 0
        return self

    def replace(self, img):
        """Set all image values within the bounding box equal to the provided img, triggers load() and imclip()"""
        self.load().imclip()
        if not (isnumpy(img) and img.shape == (self.bbox.int().height(), self.bbox.width(), self.channels()) and (img.dtype == self.array().dtype)):
            import pdb; pdb.set_trace()
        assert isnumpy(img) and img.shape == (self.bbox.int().height(), self.bbox.width(), self.channels()) and (img.dtype == self.array().dtype),  "Invalid replacement image - Must be same shape as box and same type as img"
        self.array()[int(self.bbox.ymin()):int(self.bbox.ymax()),
                     int(self.bbox.xmin()):int(self.bbox.xmax())] = img
        return self
    

class Scene(ImageCategory):
    """vipy.image.Scene class

    This class provides a representation of a vipy.image.ImageCategory with one or more vipy.object.Detections.  The goal of this class is to provide a unified representation for all objects in a scene.

    Valid constructors include all provided by vipy.image.Image() and vipy.image.ImageCategory() with the additional kwarg 'objects', which is a list of vipy.object.Detections()

    >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city', objects=[vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city').objects([vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)])
    >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh=[0,0,100,100])
    >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh=[[0,0,100,100], [100,100,200,200]])
    >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels=['face', 'desk'] xywh=[[0,0,100,100], [200,200,300,300]])

    """
    def __init__(self, filename=None, url=None, category=None, attributes=None, objects=None, xywh=None, boxlabels=None, array=None, colorspace=None):
        super(Scene, self).__init__(filename=filename, url=url, attributes=attributes, category=category, array=array, colorspace=colorspace)   # ImageCategory class inheritance
        self._objectlist = []

        if objects is not None:
            if not (isinstance(self._objectlist, list) and all([isinstance(bb, vipy.object.Detection) for bb in objects])):
                raise ValueError("Invalid object list - Input must be [vipy.object.Detection(), ...]")
            self._objectlist = objects

        detlist = []
        if xywh is not None:
            if (islistoflists(xywh) or istupleoftuples(xywh)) and all([len(bb)==4 for bb in xywh]):
                detlist = [vipy.object.Detection(category=None, xywh=bb) for bb in xywh]
            elif (islist(xywh) or istuple(xywh)) and len(xywh)==4 and all([isnumber(bb) for bb in xywh]):
                detlist = [vipy.object.Detection(category=None, xywh=xywh)]
            else:
                raise ValueError("Invalid xywh list - Input must be [[x1,y1,w1,h1], ...")            
        if boxlabels is not None:
            if isstring(boxlabels):
                label = boxlabels
                detlist = [d.category(label) for d in detlist]
            elif (istuple(boxlabels) or islist(boxlabels)) and len(boxlabels) == len(xywh):
                detlist = [d.category(label) for (d,label) in zip(detlist, boxlabels)]
            else:
                raise ValueError("Invalid boxlabels list - len(boxlabels) must be len(xywh) with corresponding labels for each xywh box  [label1, label2, ...]")

        self._objectlist = self._objectlist + detlist
        
    def __eq__(self, other):
        """Scene equality requires equality of all objects in the scene, assumes a total order of objects"""
        return isinstance(other, Scene) and len(self)==len(other) and all([obj1 == obj2 for (obj1, obj2) in zip(self, other)])

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color=%s" % (self.height(), self.width(), self.colorspace()))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl():
            strlist.append('url=%s' % self.url())
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
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
        """Return the kth object in the scene as an ImageDetection"""
        assert isinstance(k, int), "Indexing by object in scene must be integer"
        obj = self._objectlist[k]
        return (ImageDetection(array=self.array(), filename=self.filename(), url=self.url(), colorspace=self.colorspace(), bbox=obj, category=obj.category(), attributes=obj.attributes))

    def dict(self):
        d = super(Scene, self).dict()
        d['objects'] = [obj.dict() for obj in self.objects()]
        return d

    def append(self, imdet):
        """Append the provided vipy.object.Detection object to the scene object list"""
        assert isinstance(imdet, vipy.object.Detection), "Invalid input"
        self._objectlist.append(imdet)
        return self

    def objects(self, objectlist=None):
        if objectlist is None:
            return self._objectlist
        else:
            assert isinstance(objectlist, list) and all([isinstance(bb, vipy.object.Detection) for bb in objectlist]), "Invalid object list"
            self._objectlist = objectlist
            return self

    def objectmap(self, f):
        """Apply lambda function f to each object"""
        self._objectlist = [f(obj)  for obj in self._objectlist]
        assert all([isinstance(a, vipy.object.Detection) for a in self.objects()]), "Lambda function must return vipy.object.Detection"
        return self

    def objectfilter(self, f):
        """Apply lambda function f to each object and keep if filter is True"""
        self._objectlist = [obj for obj in self._objectlist if f(obj) is True]
        return self

    def nms(self, conf, iou):
        """Non-maximum supporession of objects() by category based on confidence and spatial IoU thresholds"""
        return self.objects( vipy.object.non_maximum_suppression(self.objects(), conf, iou, bycategory=True) )
    
    def boundingbox(self):
        """The boundingbox of a scene is the union of all object bounding boxes, or None if there are no objects"""
        boxes = self.objects()
        bb = boxes[0].clone() if len(boxes) >= 1 else None
        return bb.union(boxes[1:]) if len(boxes) >= 2 else bb
        
    def categories(self):
        """Return list of unique object categories in scene"""
        return list(set([obj.category() for obj in self._objectlist]))
    
    # Spatial transformation
    def imclip(self):
        """Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image"""
        self._objectlist = [bb.imclip(self.numpy()) for bb in self._objectlist if bb.hasoverlap(self.numpy())]
        return self

    def rescale(self, scale=1, interp='bilinear'):
        """Rescale image buffer and all bounding boxes - Not idemponent"""
        self = super(ImageCategory, self).rescale(scale, interp=interp)
        self._objectlist = [bb.rescale(scale) for bb in self._objectlist]
        return self

    def resize(self, cols=None, rows=None, interp='bilinear'):
        """Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly.  If cols or rows is None, then scale isotropically"""
        assert cols is not None or rows is not None, "Invalid input"
        sx = (float(cols) / self.width()) if cols is not None else None
        sy = (float(rows) / self.height()) if rows is not None else None
        sx = sy if sx is None else sx
        sy = sx if sy is None else sy        
        self._objectlist = [bb.scalex(sx).scaley(sy) for bb in self._objectlist]        
        if sx == sy:
            self = super(Scene, self).rescale(sx, interp=interp)  # FIXME: if we call resize here, inheritance is screweed up
        else:
            self = super(Scene, self).resize(cols, rows, interp=interp)
        return self

    def centersquare(self):
        """Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W)) preserving center, and update bounding boxes"""
        (H,W) = self.shape()
        self = super(ImageCategory, self).centersquare()
        (dy, dx) = ((H - self.height())/2.0, (W - self.width())/2.0)
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]
        return self
    
    def fliplr(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self._objectlist = [bb.fliplr(self.numpy()) for bb in self._objectlist]
        self = super(ImageCategory, self).fliplr()
        return self

    def dilate(self, s):
        """Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle"""
        self._objectlist = [bb.dilate(s) for bb in self._objectlist]
        return self

    def zeropad(self, padwidth, padheight):
        """Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super(ImageCategory, self).zeropad(padwidth, padheight)
        dx = padwidth[0] if isinstance(padwidth, tuple) and len(padwidth) == 2 else padwidth
        dy = padheight[0] if isinstance(padheight, tuple) and len(padheight) == 2 else padheight
        self._objectlist = [bb.translate(dx, dy) for bb in self._objectlist]
        return self

    def meanpad(self, dim):
        """Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super(ImageCategory, self).meanpad(padwidth, padheight)
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

    def maxdim(self, dim, interp='bilinear'):
        """Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects"""
        return super(ImageCategory, self).maxdim(dim, interp=interp)  # will call self.rescale() which will update boxes

    def mindim(self, dim, interp='bilinear'):
        """Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects"""
        return super(ImageCategory, self).mindim(dim, interp=interp)  # will call self.rescale() which will update boxes

    def crop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, clipping the box to the image rectangle, update all scene objects"""        
        self = super(ImageCategory, self)._crop(bbox)        
        (dx, dy) = (bbox.xmin(), bbox.ymin())
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]
        return self

    def padcrop(self, bbox):
        """Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects"""
        self.zeropad(bbox.int().width(), bbox.int().height())
        (dx, dy) = (bbox.width(), bbox.height())
        bbox = bbox.translate(dx, dy)
        self = super(ImageCategory, self)._crop(bbox)        
        (dx, dy) = (bbox.xmin(), bbox.ymin())
        self._objectlist = [bb.translate(-dx, -dy) for bb in self._objectlist]
        return self
    
    # Image export
    def rectangular_mask(self, W=None, H=None):
        """Return a binary array of the same size as the image (or using the
        provided image width and height (W,H) size to avoid an image load),
        with ones inside the bounding box"""
        if (W is None or H is None):
            (H, W) = (int(np.round(self.height())),
                      int(np.round(self.width())))
        immask = np.zeros((H, W)).astype(np.uint8)
        for bb in self._objectlist:
            if bb.hasoverlap(immask):
                bbm = bb.clone().imclip(self.numpy()).int()
                immask[bbm.ymin():bbm.ymax(), bbm.xmin():bbm.xmax()] = 1
        return immask

    def show(self, categories=None, figure=None, nocaption=False, nocaption_withstring=[], fontsize=10, boxalpha=0.25, d_category2color={'Person':'green', 'Vehicle':'blue', 'Object':'red'}, captionoffset=(0,0), nowindow=False, textfacecolor='white', textfacealpha=1.0, shortlabel=True):
        """Show scene detection 

           * categories [list]:  List of category (or shortlabel) names in the scene to show
           * fontsize (int, string): Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size
           * figure (int): Figure number, show the image in the provided figure=int numbered window
           * nocaption (bool):  Show or do not show the text caption in the upper left of the box 
           * nocaption_withstring (list):  Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list
           * boxalpha (float, [0,1]):  Set the text box background to be semi-transparent with an alpha
           * d_category2color (dict):  Define a dictionary of required mapping of specific category() to box colors.  Non-specified categories are assigned a random named color from vipy.show.colorlist()
           * caption_offset (int, int): The relative position of the caption to the upper right corner of the box.
           * nowindow (bool):  Display or not display the image
           * textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background
           * textfacealpha (float, [0,1]):  The textbox background transparency
           * shortlabel (bool):  Whether to show the shortlabel or the full category name in the caption

        """
        colors = colorlist()        
        valid_detections = [obj.clone() for obj in self._objectlist if categories is None or obj.category() in tolist(categories)]  # Detections with valid category
        valid_detections = [obj.imclip(self.numpy()) for obj in valid_detections if obj.hasoverlap(self.numpy())]  # Detections within image rectangle
        valid_detections = [obj.category(obj.shortlabel()) for obj in valid_detections] if shortlabel else valid_detections  # Display name as shortlabel?               
        valid_detections = [d if not any([c in d.category() for c in tolist(nocaption_withstring)]) else d.nocategory() for d in valid_detections]  # Detections requested to show without caption
        d_categories2color = {d.category():colors[int(hashlib.sha1(d.category().split(' ')[-1].encode('utf-8')).hexdigest(), 16) % len(colors)] for d in valid_detections}   # consistent color mapping by category suffix (space separated)
        d_categories2color.update(d_category2color)  # requested color mapping
        detection_color = [d_categories2color[d.category()] for d in valid_detections]                
        imdisplay = self.clone().rgb() if self.colorspace() != 'rgb' else self  # convert to RGB for show() if necessary
        fontsize_scaled = float(fontsize.split(':')[0])*(min(imdisplay.shape())/640.0) if isstring(fontsize) else fontsize
        vipy.show.imdetection(imdisplay._array, valid_detections, bboxcolor=detection_color, textcolor=detection_color, fignum=figure, do_caption=(nocaption==False), facealpha=boxalpha, fontsize=fontsize_scaled,
                              captionoffset=captionoffset, nowindow=nowindow, textfacecolor=textfacecolor, textfacealpha=textfacealpha)
        return self

    def savefig(self, outfile=None, categories=None, figure=None, nocaption=False, fontsize=10, boxalpha=0.25, d_category2color={'person':'green', 'vehicle':'blue', 'object':'red'}, captionoffset=(0,0), dpi=200, textfacecolor='white', textfacealpha=1.0, shortlabel=True, nocaption_withstring=[]):
        """Save show() output to given file or return bufferwithout popping up a window"""
        self.show(categories=categories, figure=figure, nocaption=nocaption, fontsize=fontsize, boxalpha=boxalpha, 
                  d_category2color=d_category2color, captionoffset=captionoffset, nowindow=True, textfacecolor=textfacecolor, 
                  textfacealpha=textfacealpha, shortlabel=shortlabel, nocaption_withstring=nocaption_withstring)
        if outfile is None:
            (W,H) = plt.gcf().canvas.get_width_height()  # fast
            buf = io.BytesIO()
            plt.gcf().canvas.print_raw(buf)  # fast
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((H, W, 4))
            plt.close(plt.gcf())   # memory cleanup
            return vipy.image.Image(array=img, colorspace='rgba')
        else:
            savefig(outfile, figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
            return outfile


def RandomImage(rows=None, cols=None):
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols
    return Image(array=np.uint8(255 * np.random.rand(rows, cols, 3)), colorspace='rgb')


def RandomImageDetection(rows=None, cols=None):
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols
    return ImageDetection(array=np.uint8(255 * np.random.rand(rows, cols, 3)), colorspace='rgb', category='RandomImageDetection',
                          xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                          bbwidth=np.random.randint(16,cols), bbheight=np.random.randint(16,rows))

def RandomScene(rows=None, cols=None):
    im = RandomImage(rows, cols)
    (rows, cols) = im.shape()
    ims = Scene(array=im.array(), colorspace='rgb', category='scene', objects=[vipy.object.Detection('obj%d' % k, xmin=np.random.randint(0,cols - 16), ymin=np.random.randint(0,rows - 16),
                                                                                                     width=np.random.randint(16,cols), height=np.random.randint(16,rows))
                                                                               for k in range(0,16)])
    return ims
    

