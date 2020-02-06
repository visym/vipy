import os
import PIL
import PIL.Image
from vipy.show import imshow, imbbox, savefig, colorlist
from vipy.util import isnumpy, quietprint, isurl, isimageurl, islist, \
    fileext, tempimage, mat2gray, imwrite, imwritejet, imwritegray, \
    tempjpg, imresize, imrescale, filetail, isimagefile, remkdir, hasextension, \
    try_import
from vipy.geometry import BoundingBox, similarity_imtransform, \
    similarity_imtransform2D, imtransform, imtransform2D
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
import matplotlib.transforms
import matplotlib.pyplot as plt
import warnings

# FIX <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
# verify failed (_ssl.c:581)>
# http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


class Image(object):
    """Vipy class for images"""
    def __init__(self, filename=None, url=None, array=None, colorspace=None, attributes=None):
        # Private attributes
        self._ignoreErrors = False  # ignore errors during fetch (broken links)
        self._urluser = None      # basic authentication set with url() method
        self._urlpassword = None  # basic authentication set with url() method
        self._urlsha1 = None      # file hash if known
        self._filename = None   # Local filename
        self._url = None        # URL to download
        self._loader = None     # lambda function to load an image, set with loader() method

        # Initialization
        self._filename = filename
        if url is not None:
            assert isurl(url), 'Invalid URL'
        self._url = url
        if array is not None:
            assert isnumpy(array), 'Invalid Array'
        self._array = array   

        # Public attributes: passed in as a dictionary
        self.attributes = attributes if attributes is not None else {'colorspace':str(None)}
        if colorspace is not None:
            assert colorspace in set(['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'grey', 'gray', 'float'])
            self.attributes['colorspace'] = colorspace
        
    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Yield single image for consistency with videos and templates"""
        yield self

    def __len__(self):
        """Images have length 1 always"""
        return 1

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], self.attributes['colorspace']))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        return str('<vipy.image: %s>' % (', '.join(strlist)))

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
                self._array = self._loader(self._filename).astype(np.float32)  # forcing float
                self.setattribute('colorspace', 'float')
            elif isimagefile(self._filename):
                self._array = np.array(PIL.Image.open(self._filename))  # RGB order!
                if self.istransparent():
                    self.setattribute('colorspace', 'rgba')  # must be before iscolor()
                elif self.iscolor():
                    self.setattribute('colorspace', 'rgb')
                elif self.isgrey():
                    self.setattribute('colorspace', 'grey')                                        
                else:
                    raise ValueError('unknown colorspace')                    
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
            
        self.flush()
        return self

    def flush(self):
        """Remove cached numpy array"""
        self._array = None
        return self

    def reload(self):
        return self.flush().load()
    
    def isloaded(self):
        return self._array is not None

    def show(self, figure=None, nowindow=False):
        """Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object"""
        assert self.load().isloaded(), 'Image not loaded'
        imshow(self.clone().rgb().numpy(), figure=figure, nowindow=nowindow)
        return self

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
        """Grey images are three channel, float32 or uint8"""        
        return self.channels() == 1
                       
    def filesize(self):
        assert self.hasfilename(), 'Invalid image filename'
        return os.path.getsize(self._filename)

    def width(self):
        return self.load().array().shape[1]

    def height(self):
        return self.load().array().shape[0]

    def shape(self):
        return (self.load().height(), self.width())

    def array(self, np_array=None):
        """Replace self._array with provided numpy array"""
        if np_array is None:
            return self._array
        elif isnumpy(np_array):
            self._array = np.copy(np_array)
            self._filename = None
            self._url = None
            return self
        else:
            raise ValueError('Invalid numpy array')

    def buffer(self, data=None):
        """Equivalent to self.array()"""
        return self.array(data)

    def tonumpy(self):
        return self.load().array()

    def numpy(self):
        return self.tonumpy()
    
    def pil(self):
        return PIL.Image.fromarray(self.tonumpy())

    def filename(self, newfile=None):
        """Image Filename"""
        if newfile is None:
            return self._filename
        else:
            # set filename and return object
            self.flush()
            self._filename = newfile
            self._url = None
            return self

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

    def uri(self):
        """Return the URI of the image object, either the URL or the filename, raise exception if neither defined"""
        if self._url is not None:
            return self._url
        elif self._filename is not None:
            return self._filename
        else:
            raise ValueError('No URI defined')

    def saveas(self, filename, writeas=None):
        """Save current buffer (not including drawing overlays) to new filename and return filename"""
        if self.attributes['colorspace'] in ['gray']:
            imwritegray(self.grayscale()._array, filename)            
        elif self.attributes['colorspace'] != 'float':
            imwrite(self.load().array(), filename, writeas=writeas)
        else:
            raise ValueError('Convert float image to RGB or gray first')
        return filename

    def saveastmp(self):
        """Save current buffer to temp JPEG filename and return filename"""
        return self.saveas(tempjpg())
    def savetmp(self):
        """Save current buffer to temp JPEG filename and return filename"""
        return self.saveas(tempjpg())


    def savefig(self, filename=None):
        """Save last figure output from self.show() with drawing overlays to provided filename and return filename"""
        self.show(figure=None, nowindow=True)
        f = filename if filename is not None else tempjpg()
        return savefig(filename=f)

    def setattribute(self, key, value):
        if self.attributes is None:
            self.attributes = {key: value}
        else:
            self.attributes[key] = value
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

    def stats(self):
        print(self)
        print('  Channels: %d' % self.channels())
        print('  Shape: %s' % str(self.shape()))
        print('  min: %d' % self.min())
        print('  max: %d' % self.max())
        print('  mean: %s' % str(self.mean()))

    def clone(self):
        """Create deep copy of image object"""
        im = copy.deepcopy(self)
        if self._array is not None:
            im._array = self._array.copy()
        return im

    def resize(self, cols=None, rows=None):
        """Resize the image buffer to (rows x cols) with bilinear interpolation.  If rows or cols is provided, rescale image maintaining aspect ratio"""
        if cols is None or rows is None:
            if cols is None:
                scale = float(rows)/float(self.height())
            else:
                scale = float(cols)/float(self.width())
            self.rescale(scale)
            
        else:
            self._array = np.array(self.load().pil().resize((cols, rows), PIL.Image.BILINEAR))

        return self

    def rescale(self, scale=1):
        """Scale the image buffer by the given factor - NOT idemponent"""
        (height, width) = self.load().shape()
        self._array = np.array(self.pil().resize((int(np.round(scale*width)), int(np.round(scale*height))), PIL.Image.BILINEAR))
        return self

    def maxside(self, dim):
        """Resize image preserving aspect ratio so that maximum dimension of bounding box = dim"""
        return self.rescale(float(dim) / float(np.maximum(self.bbox.height(), self.bbox.width())))
    
    def minside(self, dim):
        """Resize image preserving aspect ratio so that minimum dimension of bounding box = dim"""
        return self.rescale(float(dim) / float(np.minimummum(self.bbox.height(), self.bbox.width())))
        
    def maxdim(self, dim):
        """Resize image preserving aspect ratio so that maximum dimension of image = dim"""
        return self.rescale(float(dim) / float(np.maximum(self.height(), self.width())))

    def mindim(self, dim):
        """Resize image preserving aspect ratio so that minimum dimension of image = dim"""        
        return self.rescale(float(dim) / float(np.minimum(self.height(), self.width())))

    def pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode, dx=padwidth, dy=padheight"""
        self._array = np.pad(self.load().array(),
                           ((dy, dy), (dx, dx), (0, 0)) if
                           self.load().array().ndim == 3 else ((dy, dy), (dx, dx)),
                           mode=mode)
        return self

    def zeropad(self, padwidth, padheight):
        """Pad image using np.pad constant by adding dx on both left and right, and dy on top and bottom"""
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth, padwidth)
        if not isinstance(padheight, tuple):
            padheight = (padheight, padheight)            
        if self.iscolor():
            pad_shape = (padheight, padwidth, (0, 0))
        else:
            pad_shape = (padheight, padwidth)
        self._array = np.pad(self.load().array(),
                           pad_width=pad_shape,
                           mode='constant',
                           constant_values=0)        
        return self

    def meanpad(self, dx, dy):
        """Pad image using np.pad constant where constant is image mean"""
        #mu = self.mean()
        if self.load().array().ndim == 3:
            pad_size = ((dy, dy), (dx, dx), (0, 0))
        else:
            pad_size = ((dy, dy), (dx, dx))
        self._array = np.pad(self.load().array(), pad_size, mode='mean')
        return self

    def boxclip(self):
        """Clip bounding box to the image rectangle, and delete bbox if outside image rectangle.  Requires image load"""
        (H,W) = self.load().shape()
        if self.bbox.intersection(BoundingBox(xmin=0, ymin=0, xmax=W-1, ymax=H-1), strict=False).isdegenerate():
            warnings.warn("Degenerate bounding box does not intersect image - Deleting")
            self.bbox = None
        return self

    def minsquare(self):
        """Crop image of size (HxW) to (min(H,W), min(H,W))"""
        img = self.load().array()
        S = np.min(img.shape[0:2])
        self._array = self._array[0:S,0:S,:]
        return self

    def centersquare(self):
        """Crop image of size (NxN) in the center"""
        """FIXME: this is off by one"""
        n = int(np.min(self.shape()))
        return self.crop(BoundingBox(xcentroid=int(self.width()/2.0), ycentroid=int(self.height()/2.0), width=n, height=n)).resize(n,n)
        
    def maxsquare(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding"""
        S = np.max(self.shape())
        self.zeropad( (0, S-self.width()), (0, S-self.height()))
        self._array = self._array[0:S,0:S,:] if self.channels() != 1 else  self._array[0:S,0:S]
        return self

    def maxsquare_with_meanpad(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding"""
        img = self.load().array()
        S = np.max(img.shape)
        self._array = np.pad(self.load().array(), ((0,S-self.height()), (0,S-self.width()), (0,0)), mode='mean')
        self._array = self._array[0:S,0:S,:]
        return self

    def crop(self, bbox, pad='mean'):
        """Crop the image buffer using the supplied bounding box object - NOT
        idemponent."""
        assert isinstance(bbox, BoundingBox) and bbox.valid(), "Invalid bounding box"
        bbox = bbox.imclip(self.load().array())  
        self._array = self.array()[int(bbox.ymin()):int(bbox.ymax()),
                                   int(bbox.xmin()):int(bbox.xmax())]
        return self

    def fliplr(self):
        """Mirror the image buffer about the vertical axis - Not idemponent"""
        self._array = np.fliplr(self.load().array())
        return self

    def _colorspace(self, to):
        """Supported colorspaces are rgb, rgbab, bgr, bgra, hsv, grey, float"""
        self.load()
        if self.attributes['colorspace'] == to:
            return self
        elif to == 'float':
            img = self.load().array()  # any type
            self._array = np.array(img).astype(np.float32)  # typecast to float32
        elif self.attributes['colorspace'] in ['gray', 'grey']:
            img = self.load().array()  # single channel float32 [0,1] or single channel uint8 [0,255]
            if img.dtype is np.dtype(np.float32): 
                self._array = np.array(PIL.Image.fromarray(255.0*img, mode='F').convert('RGB'))  # float32 gray [0,1] -> float32 gray [0,255] -> uint8 RGB
            elif img.dtype is np.dtype(np.uint8):
                self._array = np.array(PIL.Image.fromarray(img, mode='L').convert('RGB'))  # uint8 gray [0,255] -> uint8 RGB
            else:
                raise ValueError('unsupported datatype for greyscale image')
            self.attributes['colorspace'] = 'rgb'            
            self._colorspace(to)
        elif self.attributes['colorspace'] == 'rgba':
            img = self.load().array()  # uint8 RGBA            
            if to == 'bgra':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGBA -> uint8 ABGR
                self._array = self._array[:,:,[1,2,3,0]]  # uint8 ABGR -> uint8 BGRA
            elif to == 'rgb':
                self._array = self._array[:,:,0:-1]  # uint8 RGBA -> uint8 RGB
            else:
                self._array = self._array[:,:,0:-1]  # uint8 RGBA -> uint8 RGB
                self.attributes['colorspace'] = 'rgb'
                self._colorspace(to)                          
        elif self.attributes['colorspace'] == 'rgb':
            img = self.load().array()  # uint8 RGB            
            if to in ['grey', 'gray']:
                self._array = (1.0/255.0)*np.array(PIL.Image.fromarray(img).convert('L')).astype(np.float32)  # uint8 RGB -> float32 Grey [0,255] -> float32 Grey [0,1]
            elif to == 'bgr':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGB -> uint8 BGR               
            elif to == 'hsv':
                self._array = np.array(PIL.Image.fromarray(img).convert('HSV'))  # uint8 RGB -> uint8 HSV
            elif to == 'rgba':
                self._array = np.dstack( (img, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))
            elif to == 'bgra':
                self._array = np.array(img)[:,:,::-1]  # uint8 RGB -> uint8 BGR
                self._array = np.dstack( (self._array, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))  # uint8 BGR -> uint8 BGRA
        elif self.attributes['colorspace'] == 'bgr':
            img = self.load().array()  # uint8 BGR
            self._array = np.array(img)[:,:,::-1]  # uint8 BGR -> uint8 RGB
            self.attributes['colorspace'] = 'rgb'
            self._colorspace(to)
        elif self.attributes['colorspace'] == 'bgra':
            img = self.load().array()  # uint8 BGRA
            self._array = np.array(img)[:,:,::-1]  # uint8 BGRA -> uint8 ARGB
            self._array = self._array[:,:,[1,2,3,0]]  # uint8 ARGB -> uint8 RGBA
            self.attributes['colorspace'] = 'rgba'
            self._colorspace(to)
        elif self.attributes['colorspace'] == 'hsv':
            img = self.load().array()  # uint8 HSV
            self._array = np.array(PIL.Image.fromarray(img, mode='HSV').convert('RGB'))  # uint8 HSV -> uint8 RGB
            self.attributes['colorspace'] = 'rgb'
            self._colorspace(to)
        elif self.attributes['colorspace'] == 'float':
            img = self.load().array()  # float32
            if np.max(img)>1 or np.min(img)<0:
                raise ValueError('Float image must be rescaled to the range float32 [0,1] prior to conversion')
            if not self.channels() in [1,3]:
                raise ValueError('Float image must be single channel or three channel RGB in the range float32 [0,1] prior to conversion')                       
            if self.channels() == 3:  # assumed RGB
                self._array = (1.0/255.0)*np.array(PIL.Image.fromarray(np.uint8(255*self.array())).convert('L')).astype(np.float32) # float32 RGB [0,1] -> float32 gray [0,1]
            self.attributes['colorspace'] = 'gray'            
            self._colorspace(to)
        elif self.attributes['colorspace'] is None:
            raise ValueError('Colorspace must be set during construction to allow for colorspace conversion')
        else:
            raise ValueError('unsupported colorspace "%s"' % self.attributes['colorspace'])
        self.attributes['colorspace'] = to
        return self
    
    def rgb(self):
        """Convert the image buffer to three channel RGB uint8 colorspace"""
        return self._colorspace('rgb')

    def rgba(self):
        """Convert the image buffer to four channel RGBA uint8 colorspace"""
        return self._colorspace('rgba')
    
    def hsv(self):
        """Convert the image buffer to three channel HSV uint8 colorspace"""
        return self._colorspace('hsv')

    def bgr(self):
        """Convert the image buffer to three channel BGR uint8 colorspace"""
        return self._colorspace('bgr')

    def bgra(self):
        """Convert the image buffer to four channel BGR uint8 colorspace"""
        return self._colorspace('bgra')
    
    def float(self, scale=None):
        """Convert the image buffer to float32"""
        self._colorspace('float')
        self._array = self._array * scale if scale is not None else self._array
        return self

    def grayscale(self):
        """Convert the image buffer to single channel grayscale float32 in range [0,1]"""        
        self._colorspace('gray')
        return self

    def _apply_colormap(self, cm):
        """Convert an image to greyscale, then convert to RGB image with matplotlib colormap"""
        """https://matplotlib.org/tutorials/colors/colormaps.html"""
        cm = plt.get_cmap(cm)        
        img = self.grey().numpy()
        self._array = np.uint8(255*cm(img)[:,:,:3])
        self.attributes['colorspace'] = 'rgb'
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

    def greyscale(self):
        """Equivalent to grayscale()"""
        return self.grayscale()
    def grey(self):
        """Equivalent to grayscale()"""
        return self.grayscale()
    def gray(self):
        """Equivalent to grayscale()"""
        return self.grayscale()

    def saturate(self, min, max):
        """Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type"""
        img = self.load().array()
        self._array = np.minimum(np.maximum(self.load().array(), min), max)        
        return self

    def intensity(self):
        """Convert image to float32 with [min,max] to range [0,1]"""
        self._colorspace('float')        
        self._array = (self.load().array().astype(np.float32) - float(self.min())) / float(self.max()-self.min())
        return self
    
    def min(self):
        return np.min(self.load().array().flatten())

    def max(self):
        return np.max(self.load().array().flatten())

    def mean(self):
        return np.mean(self.load().array(), axis=(0, 1)).flatten()

    def mat2gray(self, min=None, max=None):
        """Convert the image buffer so that [min,max] -> [0,1]"""
        self._colorspace('float')
        self._array = mat2gray(np.float32(self.load().array()), min, max)
        return self

    def gain(self, g):
        """Multiply gain to image array"""
        self._array = np.multiply(self.load().float().array(), g)
        return self

    def bias(self, b):
        """Add a bias to the image array"""
        self._array = self.load().float().array() + b
        return self

    def html(self, alt=None):
        im = self.clone().rgb()
        buf = io.BytesIO()
        PIL.Image.frombuffer(im).save(buf, format='JPEG')
        b = buf.getvalue().encpde('base64')

        alt_text = alt if alt is not None else im.filename()
        return '<img src="data:image/png;base64,%s" alt="%s" />' % (b, alt_text)


class ImageCategory(Image):
    def __init__(self, filename=None, url=None, category=None,
                 attributes=None, array=None, colorspace=None):
        # Image class inheritance
        super(ImageCategory, self).__init__(filename=filename,
                                            url=url,
                                            attributes=attributes,
                                            array=array,
                                            colorspace=colorspace)
        self._category = category

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], self.attributes['colorspace']))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        if self.category() is not None: 
            strlist.append('category="%s"' % self.category())
        return str('<vipy.imagecategory: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        return self._category.lower() == other._category.lower()

    def __ne__(self, other):
        return self._category.lower() != other._category.lower()

    def is_(self, other):
        return self.__eq__(other)

    def is_not(self, other):
        return self.__ne__(other)

    def __hash__(self):
        return hash(self._category.lower())

    def iscategory(self, category):
        return (self._category.lower() == category.lower())

    def ascategory(self, newcategory):
        return self.category(newcategory)

    def category(self, newcategory=None):
        if newcategory is None:
            return self._category
        else:
            self._category = newcategory
            return self

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


class ImageDetection(ImageCategory):
    def __init__(self, filename=None, url=None, category=None, attributes=None,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 bbwidth=None, bbheight=None, bbox=None, array=None, colorspace=None):
        
        # ImageCategory class inheritance
        super(ImageDetection, self).__init__(filename=filename,
                                             url=url,
                                             attributes=attributes,
                                             category=category,
                                             array=array,
                                             colorspace=colorspace)

        # Construction options
        (width,height) = (bbwidth,bbheight)
        if bbox is not None:            
            assert isinstance(bbox, BoundingBox), "Invalid bounding box"            
            self.bbox = bbox
        elif xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        elif xmin is not None and ymin is not None and width is not None and height is not None:
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin, width=width, height=height)
        else:
            # Empty box to be updated with boundingbox() method
            self.bbox = BoundingBox(xmin=0, ymin=0, width=0, height=0)
            
    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], self.attributes['colorspace']))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url="%s"' % self.url())
        if self.category() is not None: 
            strlist.append('category="%s"' % self.category())
        if self.bbox.isvalid():
            strlist.append('bbox=(xmin=%1.1f,ymin=%1.1f,xmax=%1.1f,ymax=%1.1f)' %
                           (self.bbox.xmin(), self.bbox.ymin(),self.bbox.xmax(), self.bbox.ymax()))
        return str('<vipy.imagedetection: %s>' % (', '.join(strlist)))
            
    def __hash__(self):
        return hash(self.__repr__())

    def show(self, figure=None, nowindow=False):
        self.load()
        if self.bbox.invalid():
            warnings.warn('Ignoring invalid bounding box "%s"' % str(self.bbox))                
        if self.bbox.valid() and self.bbox.hasoverlap(self.array()) and self.bbox.shape() != self._array.shape[0:2]:
            self.boxclip()  # crop bbox to image rectangle for valid overlay image
            imbbox(self.clone().rgb()._array, self.bbox.xmin(),
                   self.bbox.ymin(), self.bbox.xmax(), self.bbox.ymax(),
                   bboxcaption=self.category(),
                   fignum=figure, nowindow=nowindow)
        else:
            # Do not display the box if the box is degenerate or equal to the image rectangle
            super(ImageDetection, self).show(figure=figure, nowindow=nowindow)
        return self

    def boundingbox(self, xmin=None, xmax=None, ymin=None, ymax=None,
                    bbox=None, width=None, height=None, dilate=None,
                    xcentroid=None, ycentroid=None, dilate_topheight=None):        
        if xmin is None and xmax is None and \
           ymin is None and ymax is None and \
           bbox is None and \
           width is None and height is None and \
           dilate is None and dilate_topheight is None:
            return self.bbox
        elif (xmin is not None and xmax is not None and
              ymin is not None and ymax is not None):
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin,
                                    xmax=xmax, ymax=ymax)
        elif bbox is not None:
            self.bbox = bbox
        elif (xmin is not None and ymin is not None and
              width is not None and height is not None):
            self.bbox = BoundingBox(xmin=xmin, ymin=ymin,width=width,height=height)
        elif (xcentroid is not None and ycentroid is not None and
              width is not None and height is not None):
            self.bbox = BoundingBox(xcentroid=xcentroid,
                                    ycentroid=ycentroid,
                                    width=width,
                                    height=height)
        elif (dilate is None and dilate_topheight is None):
            raise ValueError('Invalid bounding box - Either rect coordinates '
                             'or bbox object must be provided')

        if dilate_topheight is not None:
            self.bbox.dilate_topheight(dilate_topheight)
        if dilate is not None:
            self.bbox.dilate(dilate)
        return self


    def invalid(self, flush=False):
        return self.bbox.invalid() or not super(ImageDetection, self).isvalid(
            flush=flush)

    def isinterior(self, W=None, H=None, flush=False):
        """Is the bounding box fully within the image rectangle?  Use provided
        image width and height (W,H) to avoid lots of reloads in some
        conditions"""
        (W, H) = (W if W is not None else self.width(),
                  H if H is not None else self.height())
        if flush:
            self.flush()
        return (self.bbox.xmin() >= 0 and self.bbox.ymin() >= 0 and
                self.bbox.xmax() < W and self.bbox.ymax() < H)

    def rescale(self, scale=1):
        """Rescale image buffer and bounding box - Not idemponent"""
        self = super(ImageDetection, self).rescale(scale)
        self.bbox = self.bbox.rescale(scale)
        return self


    def resize(self, cols=None, rows=None):
        """Resize image buffer and bounding box"""
        self = super(ImageDetection, self).resize(cols, rows)
        self.bbox = BoundingBox(xmin=0, ymin=0, xmax=self.width(), ymax=self.height())
        return self

    def fliplr(self):
        """Mirror buffer and bounding box around vertical axis"""
        self = super(ImageDetection, self).fliplr() 
        xmin = self.bbox.xmin()
        xmax = self.bbox.xmax()
        self.bbox._xmin = self.width() - xmax
        self.bbox._xmax = self.bbox.xmin() + (xmax-xmin)
        return self

    def crop(self, bbox=None):
        """Crop image and update bounding box"""
        if bbox is not None:
            assert isinstance(bbox, BoundingBox), "Invalid bounding box"                    
        self = super(ImageDetection, self).crop(
            bbox=self.bbox if bbox is None else bbox)
        if self.bbox is not None:
            self.bbox = BoundingBox(xmin=0, ymin=0,
                                    xmax=self.width(), ymax=self.height())
        return self

    def centercrop(self, bbwidth, bbheight):
        (W,H) = (self.width(), self.height())
        self.bbox = BoundingBox(xcentroid=W/2.0, ycentroid=H/2.0, width=bbwidth, height=bbheight)
        return self.crop()

    def maxsquare(self):
        """Return a square bounding box centered at current centroid"""
        self.bbox = self.bbox.maxsquare()
        return self

    def dilate(self, s):
        """Dilate bounding box by scale factor"""
        self.bbox = self.bbox.dilate(s)
        return self

    def pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode"""
        # image class inheritance
        self = super(ImageDetection, self).pad(dx, dy, mode)
        self.bbox = self.bbox.translate(dx, dy)
        return self

    def zeropad(self, dx, dy):
        """Pad image using np.pad constant"""
        # image class inheritance
        self = super(ImageDetection, self).zeropad(dx, dy)
        self.bbox = self.bbox.translate(dx, dy)
        return self

    def meanpad(self, dx, dy):
        """Pad image using np.pad constant where constant is the RGB mean per
        image"""
        # image class inheritance
        self = super(ImageDetection, self).meanpad(dx, dy)
        self.bbox = self.bbox.translate(dx, dy)
        return self

    def mask(self, W=None, H=None):
        """Return a binary array of the same size as the image (or using the
        provided image width and height (W,H) size to avoid an image load),
        with ones inside the bounding box"""
        if (W is None or H is None):
            (H, W) = (int(np.round(self.height())),
                      int(np.round(self.width())))
        immask = np.zeros((H, W)).astype(np.uint8)
        (ymin, ymax, xmin, xmax) = (int(np.round(self.bbox.ymin())),
                                    int(np.round(self.bbox.ymax())),
                                    int(np.round(self.bbox.xmin())),
                                    int(np.round(self.bbox.xmax())))
        immask[ymin:ymax, xmin:xmax] = 1
        return immask

    def setzero(self, bbox=None):
        """Set all image values within the bounding box to zero"""
        if bbox is not None:
            assert isinstance(bbox, BoundingBox), "Invalid bounding box"                        
        bbox = self.bbox if bbox is None else bbox
        self.load().array()[int(bbox.ymin()):int(bbox.ymax()),
                    int(bbox.xmin()):int(bbox.xmax())] = 0
        return self


class Scene(ImageCategory):
    """A scene is an ImageCategory with one or more object detections"""
    def __init__(self, filename=None, url=None, category=None, attributes=None, objects=None, array=None, colorspace=None):
        super(Scene, self).__init__(filename=filename, url=url, attributes=attributes, category=category, array=array, colorspace=colorspace)   # ImageCategory class inheritance        
        self._objectlist = []

        if objects is not None:
            if not (isinstance(self._objectlist, list) and all([isinstance(bb, vipy.object.Detection) for bb in objects])):
                raise ValueError("Invalid object list")
            self._objectlist = objects

    def __repr__(self):
        strlist = []
        if self.isloaded():
            strlist.append("height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], self.attributes['colorspace']))
        if self.hasfilename():
            strlist.append('filename="%s"' % self.filename())
        if self.hasurl(): 
            strlist.append('url=%s' % self.url())
        if self.category() is not None:
            strlist.append('category=%s' % self.category())
        if len(self.objects()) > 0:
            strlist.append('objects=%d' % len(self.objects()))
        return str('<vipy.image.scene: %s>' % (', '.join(strlist)))

    def __len__(self):
        return len(self._objectlist)

    def __iter__(self):        
        for im in self._objectlist:
            yield im
    
    def __getitem__(self, k):        
        return self._objectlist[k]

    def rescale(self, scale=1):
        """Rescale image buffer and all bounding boxes - Not idemponent"""
        self = super(ImageCategory, self).rescale(scale)
        self._objectlist = [bb.rescale(scale) for bb in self._objectlist]
        return self

    def resize(self, cols=None, rows=None):
        """Resize image buffer and all bounding boxes"""
        sx = (float(cols)/self.width()) if cols is not None else 1.0
        sy = (float(rows)/self.height()) if rows is not None else 1.0
        self = super(ImageCategory, self).resize(cols, rows)
        self._objectlist = [bb.scalex(sx).scaley(sy) for bb in self._objectlist]
        print(self._objectlist)
        return self

    def fliplr(self):
        """Mirror buffer and all bounding box around vertical axis"""
        self = super(ImageDetection, self).fliplr()
        self._objectlist = [bb.fliplr(self.numpy()) for bb in self._objectlist]        
        return self

    def dilate(self, s):
        """Dilate all bounding boxes by scale factor"""
        self._objectlist = [bb.dilate(s) for bb in self._objectlist]                
        return self

    def zeropad(self, padwidth, padheight):
        """Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets"""
        self = super(ImageCategory, self).zeropad(padwidth, padheight)
        self._objectlist = [bb.translate(padwidth, padheight) for bb in self._objectlist]
        return self

    def mask(self, W=None, H=None):
        """Return a binary array of the same size as the image (or using the
        provided image width and height (W,H) size to avoid an image load),
        with ones inside the bounding box"""
        if (W is None or H is None):
            (H, W) = (int(np.round(self.height())),
                      int(np.round(self.width())))
        immask = np.zeros((H, W)).astype(np.uint8)
        for bb in self._objectlist:
            (ymin, ymax, xmin, xmax) = (int(np.round(bb.ymin())),
                                        int(np.round(bb.ymax())),
                                        int(np.round(bb.xmin())),
                                        int(np.round(bb.xmax())))
            immask[ymin:ymax, xmin:xmax] = 1
        return immask
    
    def append(self, imdet):
        assert isinstance(imdet, Detection), "Invalid input"
        self._objectlist.append(imdet)
        return self
    
    def show(self, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), nowindow=False):
        """Show scene detection with an optional subset of categories"""        
        valid_categories = sorted(self.categories() if category is None else tolist(category))
        valid_detections = [obj for obj in self._objectlist if obj.category() in valid_categories]
        valid_detections = [obj.imclip(self.numpy()) for obj in self._objectlist if obj.hasoverlap(self.numpy())]
        if categoryColor is None:
            colors = colorlist()
            categoryColor = dict([(c, colors[k]) for (k, c) in enumerate(valid_categories)])
        detection_color = [categoryColor[im.category()] for im in valid_detections]
        vipy.show.imdetection(self.clone().rgb()._array, valid_detections, bboxcolor=detection_color, textcolor=detection_color, fignum=figure, do_caption=do_caption, facealpha=boxalpha, fontsize=fontsize, captionlist=captionlist, captionoffset=captionoffset, nowindow=nowindow)
        return self

    def savefig(self, outfile=None, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), dpi=200):
        """Show a subset of object categores in current image and save to the given file"""
        outfile = outfile if outfile is not None else tempjpg()
        self.show(category, figure, do_caption, fontsize, boxalpha, captionlist, categoryColor, captionoffset, nowindow=True)
        savefig(outfile, figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
        return outfile

    def objects(self, objectlist=None):
        if objectlist is None:
            return self._objectlist
        else:
            assert isinstance(objectlist, list) and all([isinstance(bb, vipy.object.Detection) for bb in objectlist]), "Invalid object list"                        
            s = self.clone()
            s._objectlist = objectlist
            return s

    def clone(self):
        return deepcopy(self)
    
    def categories(self):
        return list(set([obj.category() for obj in self._objectlist]))
    
    
class Batch(object):
    def __init__(self, imlist):        
        """Create a batch of homogeneous vipy.image objects that can be operated on with a single function call"""
        assert isinstance(imlist, list) and all([isinstance(im, Image) for im in imlist]), "Invalid input"
        self._imlist = imlist

    def __repr__(self):
        return '<vipy.image.Batch: type="%s", batchsize=%d>' % (type(self._imlist[0]), len(self))

    def __len__(self):
        return len(self._imlist)
    
    def __getitem__(self, k):
        return self._imlist[k]
    
    def list(self, imlist_=None):
        if imlist_ is None:
            return self._imlist
        else:
            self._imlist = imlist_
        return self
        
    def __getattr__(self, attr):
        """Call the same method on all Image objects.  The called method must return the image object."""
        assert hasattr(self._imlist[0], attr), "Invalid attribute"
        assert attr not in set(['show', 'saveas', 'savefig', 'mask']), "Invalid attribute"
        assert attr[0:2] != 'is' and attr[0:3] != 'has', "Invalid attribute"
        return lambda *args, **kw: self.list([getattr(im,attr)(*args, **kw) for im in self._imlist])

    def torch(self):
        try_import('torch');  import torch
        
        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        return torch.from_numpy(np.vstack([np.expand_dims(im.rgb().numpy(),0) for im in self._imlist]).transpose(0,3,1,2))
    
def RandomImage(rows=None, cols=None):
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols         
    return Image(array=np.uint8(255*np.random.rand(rows, cols, 3)), colorspace='rgb')

def RandomImageDetection(rows=None, cols=None):
    rows = np.random.randint(128, 1024) if rows is None else rows
    cols = np.random.randint(128, 1024) if cols is None else cols     
    return ImageDetection(array=np.uint8(255*np.random.rand(rows, cols, 3)), colorspace='rgb', category='RandomImageDetection',
                          xmin=np.random.randint(0,cols), ymin=np.random.randint(0,rows),
                          bbwidth=np.random.randint(16,cols), bbheight=np.random.randint(16,rows))

        
        
