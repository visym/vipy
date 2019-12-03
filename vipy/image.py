import os
import PIL
from vipy.show import imshow, imbbox, savefig
from vipy.util import isnumpy, quietprint, isurl, isimageurl, islist, \
    fileext, tempimage, mat2gray, imwrite, imwritejet, imwritegray, tmpjpg, tempjpg, imresize, imrescale, bgr2gray, gray2rgb, bgr2rgb, rgb2bgr, gray2hsv, bgr2hsv
from vipy.geometry import BoundingBox, similarity_imtransform, \
    similarity_imtransform2D, imtransform, imtransform2D
import vipy.dataset.download
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib
#import cv2
import copy
import numpy as np
import shutil
import io



# FIX <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
# verify failed (_ssl.c:581)>
# http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


class Image(object):
    """Standard class for images"""
    def __init__(self, filename=None, url=None, array=None, attributes=None):
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
        self.attributes = attributes  

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Yield single image for consistency with videos and templates"""
        yield self

    def __len__(self):
        """Images have length 1 always"""
        return 1

    def __repr__(self):
        if self.isloaded():
            return str('<vipy.image: height=%d, width=%d, color=%s>' % (
                self._array.shape[0], self._array.shape[1],
                str(self.getattribute('colorspace'))))
        elif self._filename is not None and self._url is None:
            return str("<vipy.image: filename='%s'>" % str(self._filename))
        elif self._url is not None and self._filename is None:
            return str("<vipy.image: url='%s'>" % str(self._url))
        elif self._url is not None and self._filename is not None:
            return str("<vipy.image: url='%s', filename='%s'>" % (str(self._url), str(self._filename)))
        else:            
            raise ValueError('vipy.image.Image must be constructed with an input')  # should never get here

    def loader(self, f):
        """Lambda function to load an image filename to a numpy array"""
        self._loader = f
        return self


    def load(self, ignoreErrors=False, asRaw=False):
        """Load image to cached private '_array' attribute and return numpy array"""
        try:
            # Return previously loaded image
            if self._array is not None:
                return self._array

            # Download URL to filename
            if self._url is not None:
                if self._filename is None and isimageurl(self._url):
                    self._filename = tempimage(fileext(self._url))
                elif self._filename is None:
                    self._filename = tempjpg()  # guess .jpg
                self.download(ignoreErrors=ignoreErrors)

            # Load filename to numpy array
            if fileext(self._filename) == '.npz':
                if self._loader is None:
                    raise ValueError('Must define a customer loader for '
                                     '.npz file format')
                self._array = self._loader(self._filename)
            else:
                # BGR color order!
                self._array = np.array(PIL.Image.open(self._filename))[:,:,::-1]  
                #self._array = cv2.imread(self._filename,
                #                       cv2.CV_LOAD_IMAGE_UNCHANGED) \
                #                       if asRaw else cv2.imread(self._filename)

            if self._array is None:
                if fileext(self._filename) == '.gif':
                    #quietprint('[vipy.image][WARNING]: IO error - could '
                    #           'not load "%s" using opencv, '
                    #           'falling back on PIL ' %
                    #           self._filename, 1)
                    # Convert .gif to luminance (grayscale) and export
                    # as numpy array
                    self._array = np.array(
                        PIL.Image.open(self._filename).convert('L'))
                else:
                    raise ValueError('invalid image file "%s"' % self._filename)

            # Image Atributes
            self.setattribute('colorspace',
                              'bgr' if self._array.ndim == 3 else 'gray')

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[vipy.image][WARNING]: IO error - '
                           'Invalid image file, url or invalid write '
                           'permissions "%s" ' %
                           self.filename(), True)
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[vipy.image][WARNING]: '
                           'load error for image "%s"' %
                           self.filename(), verbosity=2)
                self._array = None
            else:
                raise

        return self._array

    def download(self, ignoreErrors=False, timeout=10):
        """Download URL to filename provided by constructor, or to temp filename"""
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[vipy.image.download][ERROR]: '
                             'Invalid URL "%s" ' % self._url)
        if self._filename is None:
            self._filename = tempimage(fileext(self._url))
        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if url_scheme in ['http', 'https']:
                #quietprint('[vipy.image.download]: '
                #           'downloading "%s" to "%s" ' %
                #           (self._url, self._filename), verbosity=1)
                vipy.dataset.download.download(self._url,
                                        self._filename,
                                        verbose=False,
                                        timeout=timeout,
                                        sha1=self._urlsha1,
                                        username=self._urluser,
                                        password=self._urlpassword)
            elif url_scheme == 'file':
                #quietprint('[vipy.image.download]: copying "%s" to "%s" ' %
                #           (self._url, self._filename), verbosity=2)
                shutil.copyfile(self._url, self._filename)
            elif url_scheme == 'hdfs':
                raise NotImplementedError('FIXME: support for '
                                          'hadoop distributed file system')
            else:
                raise NotImplementedError(
                    'Invalid URL scheme "%s" for URL "%s"' %
                    (url_scheme, self._url))

        except (httplib.BadStatusLine,
                urllib.error.URLError,
                urllib.error.HTTPError):
            if self._ignoreErrors or ignoreErrors:
                quietprint('[vipy.image][WARNING]: download failed - '
                           'ignoring image', 1)
                self._array = None
            else:
                raise

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[vipy.image][WARNING]: IO error - '
                           'Invalid image file, url or '
                           'invalid write permissions "%s" ' %
                           self.filename(), True)
                self._array = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[vipy.image][WARNING]: '
                           'load error for image "%s"' %
                           self.filename(), verbosity=2)
            else:
                raise
        self.flush()
        return self

    def flush(self):
        """Remove cached numpy array"""
        self._array = None
        return self

    def isloaded(self):
        return self._array is not None

    def show(self, colormap=None, figure=None, flip=True):
        assert self.isloaded(), 'Image not loaded'
        #quietprint('[vipy.image][%s]: displaying image' %
        #           (self.__repr__()), verbosity=2)
        if self.iscolor():
            if colormap == 'gray':
                imshow(self.clone().grayscale().rgb()._array, figure=figure)
            else:
                imshow(self.clone().rgb()._array, figure=figure)
        else:
            imshow(self._array, colormap=colormap,
                   figure=figure, do_updateplot=flip)
        return self

    def imagesc(self):
        imshow(self.clone().mat2gray().rgb()._array)
        return self

    def mediatype(self):
        return 'image'

    def iscolor(self):
        return self.load().ndim == 3

    def filesize(self):
        assert self.hasfilename(), 'Invalid image filename'
        return os.path.getsize(self._filename)

    def width(self):
        return self.load().shape[1]

    def height(self):
        return self.load().shape[0]

    def shape(self):
        return self.tonumpy().shape

    def array(self, data=None):
        if data is None:
            return self._array
        elif isnumpy(data):
            self._array = np.copy(data)
            self._filename = None
            self._url = None
            return self
        else:
            raise ValueError('Invalid numpy array')

    def buffer(self, data=None):
        return self.array(data)

    def tonumpy(self):
        return self.array()

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
        if self.load().ndim == 3:
            imwrite(self.load(), filename, writeas=writeas)
        else:
            imwritegray(self.grayscale()._array, filename)
        self.flush()
        self._filename = filename
        return self

    def savetmp(self):
        return self.saveas(tempjpg())

    def savefig(self, filename=None):
        f = filename if filename is not None else tmpjpg()
        savefig(filename=f)
        return f

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
        x = self._array.flatten()
        print('Channels: %d' % len(self._array))
        print('Shape: %s' % str(self._array.shape))
        print('min: %d' % np.min(x))
        print('max: %d' % np.max(x))
        print('mean: %d' % np.mean(x))
        print('std: %d' % np.std(x))

    # MODIFY IMAGE ---------------------------------------------------------
    def clone(self):
        """Create deep copy of image object"""
        im = copy.deepcopy(self)
        if self._array is not None:
            im._array = self._array.copy()
        return im

    def resize(self, cols=None, rows=None):
        """Resize the image buffer to (rows x cols)"""
        if cols is None or rows is None:
            if cols is None:
                scale = float(rows)/float(self.height())
            else:
                scale = float(cols)/float(self.width())
            #quietprint('[vipy.image][%s]: scale=%1.2f' %
            #           (self.__repr__(), scale), verbosity=2)

            # OpenCV decimation introduces artifacts using cubic
            # interp, INTER_AREA is recommended according to the
            # OpenCV docs
            #interp_method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC            
            #self._array = cv2.resize(self.load(), dsize=(0, 0),
            #                       fx=scale, fy=scale,
            #                       interpolation=interp_method)
            self._array = imrescale(self.load(), scale)
            
        else:
            #quietprint('[vipy.image][%s]: resize=(%d,%d)' %
            #           (self.__repr__(), rows, cols), verbosity=2)
            try:
                #interp_method = cv2.INTER_AREA if (
                #    rows < self.height() or
                #    cols < self.width()) else cv2.INTER_CUBIC
                # fixed bug since opencv takes x and y not rows, cols
                #self._array = cv2.resize(self.load(), dsize=(cols, rows),
                #                       interpolation=interp_method)
                self._array = imresize(self.load(), rows, cols)

            except:
                print(self)  # DEBUGGING
                raise

        dtype = self._array.dtype
        if dtype == np.float32 or dtype == np.float64:
            np.clip(self._array, 0.0, 1.0, out=self._array)
        return self

    def rescale(self, scale=1):
        """Scale the image buffer by the given factor - NOT idemponent"""
        #quietprint('[vipy.image][%s]: scale=%1.2f to (%d,%d)' %
        #           (self.__repr__(), scale, scale*self.width(),
        #            scale*self.height()), verbosity=2)

        # OpenCV decimation introduces artifacts using cubic interp ,
        # INTER_AREA is recommended according to the OpenCV docs
        #interp_method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        #self._array = cv2.resize(self.load(), dsize=(0, 0),
        #                       fx=scale, fy=scale, interpolation=interp_method)
        self._array = imrescale(self.load(), scale)

        dtype = self._array.dtype
        if dtype == np.float32 or dtype == np.float64:
            np.clip(self._array, 0.0, 1.0, out=self._array)
        return self

    def maxdim(self, dim):
        return self.rescale(float(dim) / float(np.maximum(self.height(), self.width())))

    def mindim(self, dim):
        return self.rescale(float(dim) / float(np.minimum(self.height(), self.width())))

    def pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode"""
        self._array = np.pad(self.load(),
                           ((dx, dx), (dy, dy), (0, 0)) if
                           self.load().ndim == 3 else ((dx, dx), (dy, dy)),
                           mode=mode)
        return self

    def zeropad(self, dx, dy):
        """Pad image using np.pad constant"""
        if self.load().ndim == 3:
            pad_width = ((dx, dx), (dy, dy), (0, 0))
        else:
            pad_width = ((dx, dx), (dy, dy))
        self._array = np.pad(self.load(),
                           pad_width=pad_width,
                           mode='constant',
                           constant_values=0)        
        return self

    def meanpad(self, dx, dy):
        """Pad image using np.pad constant where constant is image mean"""
        #mu = self.mean()
        if self.load().ndim == 3:
            pad_size = ((dx, dx), (dy, dy), (0, 0))
            #constant_values = tuple([(x, y) for (x, y) in zip(mu, mu)])
        else:
            #constant_values = ((mu, mu), (mu, mu))
            pad_size = ((dx, dx), (dy, dy))
        self._array = np.pad(self.load(), pad_size, mode='mean')
        return self


    def minsquare(self):
        """Crop image of size (HxW) to (min(H,W), min(H,W))"""
        img = self.load()
        S = np.min(img.shape[0:2])
        self._array = self._array[0:S,0:S,:]
        return self

    def maxsquare(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding"""
        img = self.load()
        S = np.max(img.shape)
        self._array = np.pad(self.load(), ((0,S-self.height()), (0,S-self.width()), (0,0)), mode='constant')
        self._array = self._array[0:S,0:S,:]
        return self

    def maxsquare_with_meanpad(self):
        """Crop image of size (HxW) to (max(H,W), max(H,W)) with zeropadding"""
        img = self.load()
        S = np.max(img.shape)
        self._array = np.pad(self.load(), ((0,S-self.height()), (0,S-self.width()), (0,0)), mode='mean')
        self._array = self._array[0:S,0:S,:]
        return self

    def crop(self, bbox=None, pad='mean'):
        """Crop the image buffer using the supplied bounding box - NOT
        idemponent."""
        if bbox is not None:
            if islist(bbox):
                bbox = BoundingBox(xmin=bbox[0], ymin=bbox[1],
                                   xmax=bbox[2], ymax=bbox[3])

            bbox = bbox.imclip(self.load())  # FIXME
            #quietprint('[vipy.image][%s]: cropping "%s"' %
            #           (self.__repr__(), str(bbox)), verbosity=2)
            bbox = bbox.imclip(self.load())
            # assumed numpy
            self._array = self.load()[int(bbox.ymin):int(bbox.ymax),
                                    int(bbox.xmin):int(bbox.xmax)]
        return self

    def fliplr(self):
        """Mirror the image buffer about the vertical axis - Not idemponent"""
        quietprint('[vipy.image][%s]: fliplr' %
                   (self.__repr__()), verbosity=2)
        self._array = np.fliplr(self.load())
        return self


    def normalize(self):
        """Convert image to float32 with [min,max] = [0,1]"""
        self._array = (self.load().astype(np.float32) - float(self.min())) / float(self.max()-self.min())
        return self

    def raw(self, normalized=True):
        """Load the image as a raw image buffer"""
        quietprint('[vipy.image][%s]: loading raw imagery data' %
                   (self.__repr__()), verbosity=2)
        self._array = self.load(asRaw=True)
        return self

    def grayscale(self):
        """Convert the image buffer to grayscale"""
        if self.load().ndim == 3:
            #quietprint('[vipy.image][%s]: converting to grayscale' %
            #           (self.__repr__()), verbosity=3)
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_BGR2GRAY)
            self._array = bgr2gray(self.load())
            self.setattribute('colorspace', 'gray')
        return self

    def greyscale(self):
        """Convert the image buffer to grayscale"""
        return self.grayscale()

    def rgb(self):
        """Convert the image buffer to RGB"""
        if self.load().ndim == 3:
            #quietprint('[vipy.image][%s]: converting bgr to rgb' %
            #           (self.__repr__()), verbosity=2)
            # opencv BGR to RGB
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_BGR2RGB)
            self._array = bgr2rgb(self.load())
        elif self.load().ndim == 2:
            #quietprint('[vipy.image][%s]: converting gray to rgb' %
            #           (self.__repr__()), verbosity=2)
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_GRAY2RGB)
            self._array = gray2rgb(self.load())
        self.setattribute('colorspace', 'rgb')
        return self

    def hsv(self):
        """Convert the image buffer to HSV color space"""
        if self.iscolor():
            #quietprint('[vipy.image][%s]: converting to hsv' %
            #           (self.__repr__()), verbosity=2)
            # opencv BGR (assumed) to HSV
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_BGR2HSV)
            self._array = bgr2hsv(self.load())
            self.setattribute('colorspace', 'hsv')
        else:
            #quietprint('[vipy.image][%s]: converting grayscale to hsv' %
            #           (self.__repr__()), verbosity=2)
            # grayscale -> RGB -> HSV (HACK)
            #self._array = cv2.cvtColor(self.rgb().load(), cv2.COLOR_RGB2HSV)
            self._array = gray2hsv(self.load())
        return self

    def bgr(self):
        """Convert the image buffer to BGR color format"""
        if self.load().ndim == 3:
            # Loaded by opencv, so assumed BGR order
            #self._array = self.load()
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_RGB2BGR)
            self._array = self.load()[:,:,::-1]
        elif self.load().ndim == 2:
            #quietprint('[vipy.image][%s]: converting gray to bgr' %
            #           (self.__repr__()), verbosity=2)
            #self._array = cv2.cvtColor(self.load(), cv2.COLOR_GRAY2BGR)
            self._array = gray2bgr(self.load())
        self.setattribute('colorspace', 'bgr')
        return self

    def float(self, scale=None):
        """Convert the image buffer to float32"""
        if self.load().dtype != np.float32:
            #quietprint('[vipy.image][%s]: converting to float32' %
            #           (self.__repr__()), verbosity=2)
            self._array = np.float32(self.load())
        if scale is not None:
            self._array = self._array * scale
        return self

    def uint8(self, scale=None):
        """Convert the image buffer to uint8"""
        if scale is not None:
            self._array = self.load() * scale
        if self.load().dtype != np.uint8:
            #quietprint('[vipy.image][%s]: converting to uint8' %
            #           (self.__repr__()), verbosity=2)
            self._array = np.uint8(self.load())
        return self

    def preprocess(self, scale=1.0/255.0):
        """Preprocess the image buffer by converting to grayscale, convert to
        float, then rescaling [0,255], [0,1] - Not idemponent."""
        quietprint('[vipy.image][%s]: preprocessing' %
                   (self.__repr__()), verbosity=2)
        self = self.grayscale().float()
        self._array = scale*self._array
        return self

    def min(self):
        return np.min(self.load().flatten())

    def max(self):
        return np.max(self.load().flatten())

    def mean(self):
        return np.mean(self.load(), axis=(0, 1)).flatten()

    def mat2gray(self, min=None, max=None):
        """Convert the image buffer so that [min,max] -> [0,1]"""
        quietprint('[vipy.image][%s]: contrast equalization' %
                   (self.__repr__()), verbosity=2)
        self._array = mat2gray(np.float32(self.load()), min, max)
        return self

    def transform2D(self, txy=(0, 0), r=0, s=1):
        """Transform the image buffer using a 2D similarity transform - Not
        idemponent."""
        quietprint('[vipy.image][%s]: transform2D' %
                   (self.__repr__()), verbosity=2)
        self.load()
        c = (self._array.shape[1] / 2, self._array.shape[0] / 2)
        M = similarity_imtransform2D(c=c, r=r, s=s)
        self._array = imtransform2D(self._array, M)
        A = similarity_imtransform(txy=txy)
        self._array = imtransform(self._array, A)
        return self

    def transform(self, A):
        """Transform the image buffer using the supplied affine transformation
        - Not idemponent."""
        quietprint('[vipy.image][%s]: transform' %
                   (self.__repr__()), verbosity=2)
        self._array = imtransform(self.load(), A)
        return self

    def gain(self, g):
        self._array = np.multiply(self.load(), g)
        return self

    def bias(self, b):
        self._array = self.load() + b
        return self

    def imrange(self):
        self._array = np.minimum(np.maximum(self.load(), 0), 255)
        return self

    def map(self, f):
        """Transform the image object using the supplied function - May not be
        idemponent."""
        return f(self)

    def drawbox(self, bbox, border=None, color=None, alpha=None, beta=None):
        self.load()
        dtype = self._array.dtype

        border = 2 if border is None else border
        alpha = 1.5 if alpha is None else alpha
        beta = 0.10 if dtype == np.float32 and beta is None else beta
        beta = 15 if beta is None else beta

        xmin = int(round(max(0, bbox.xmin-border)))
        ymin = int(round(max(0, bbox.ymin-border)))
        xmax = int(round(min(self.width(), bbox.xmax+border)))
        ymax = int(round(min(self.height(), bbox.ymax+border)))

        if dtype == np.float32:
            color = color if color is not None else 0.0
            clip = 1.0
        else:
            color = color if color is not None else (0, 196, 0)
            clip = 255

        data = self._array.astype(np.float32)
        data[ymin:ymax, xmin:xmax] *= alpha
        data[ymin:ymax, xmin:xmax] += beta

        data = np.clip(data, 0, clip)
        data = data.astype(dtype)

        self._array[ymin:ymin+border, xmin:xmax] = color
        self._array[ymax-border:ymax, xmin:xmax] = color
        self._array[ymin:ymax, xmin:xmin+border] = color
        self._array[ymin:ymax, xmax-border:xmax] = color

        return self

    def html(self, alt=None):
        im = self.clone().rgb()
        #ret, data = cv2.imencode('.png', im._array)
        #b = data.tobytes().encode('base64')

        buf = io.BytesIO()
        PIL.Image.frombuffer(im).save(buf, format='JPEG')
        b = buf.getvalue().encpde('base64')

        alt_text = alt if alt is not None else im.filename()
        return '<img src="data:image/png;base64,%s" alt="%s" />' % (b,
                                                                    alt_text)


class ImageCategory(Image):
    def __init__(self, filename=None, url=None, category=None,
                 attributes=None, array=None):
        # Image class inheritance
        super(ImageCategory, self).__init__(filename=filename,
                                            url=url,
                                            attributes=attributes,
                                            array=array)
        self._category = category

    def __repr__(self):
        if self.isloaded():
            str_size = ", height=%d, width=%d, color='%s'" % (
                self._array.shape[0], self._array.shape[1],
                str(self.getattribute('colorspace')))
        else:
            str_size = ""
        if self._url is None:
            if self._filename is None:
                str_file = ""
            else:
                str_file = "filename='%s'" % str(self._filename)
        else:
            str_file = "url='%s', filename='%s'" % (
                str(self._url), str(self._filename))

        str_category = ", category='%s'" % self._category
        return str('<vipy.imagecategory: %s%s%s>' % (str_file,
                                                      str_category, str_size))

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
    def __init__(self, filename=None, url=None, category=None,
                 attributes=None, xmin=float('nan'),
                 xmax=float('nan'), ymin=float('nan'), ymax=float('nan'), width=None, height=None,
                 bbox=None):
        # ImageCategory class inheritance
        super(ImageDetection, self).__init__(filename=filename,
                                             url=url,
                                             attributes=attributes,
                                             category=category)

        if bbox is not None:
            self.bbox = bbox
        elif width is None and height is None:
            self.bbox = BoundingBox(xmin=float(xmin), ymin=float(ymin),
                                    xmax=float(xmax), ymax=float(ymax))
        elif width is not None and height is not None:
            self.bbox = BoundingBox(xmin=float(xmin), ymin=float(ymin),
                                    width=float(width), height=float(height))
        else:
            raise ValueError('invalid parameterization')
            
    def __repr__(self):
        if self.isloaded():
            str_size = ", height=%d, width=%d, color='%s'" % (
                self._array.shape[0],
                self._array.shape[1],
                self.getattribute('colorspace'))
        else:
            str_size = ""
        str_category = "category='%s'" % self.category()
        str_detection = (", bbox=(xmin=%1.1f,ymin=%1.1f,"
                         "xmax=%1.1f,ymax=%1.1f)" % (
                             self.bbox.xmin, self.bbox.ymin,
                             self.bbox.xmax, self.bbox.ymax))
        str_url = "url='%s', " % str(self._url)
        if self._url is not None:
            str_file = ", url='%s', filename='%s'" % (str(self._url),
                                                      str(self._filename))
        else:
            if self.filename is None:
                str_file = ''
            else:
                str_file = ", filename='%s'" % (str(self._filename))

        return str('<vipy.imagedetection: %s%s%s%s>' % (
            str_category, str_detection, str_file, str_size))

    def __hash__(self):
        return hash(self.__repr__())

    def show(self, ignoreErrors=False, colormap=None, figure=None, flip=True):
        if self.load(ignoreErrors=ignoreErrors) is not None:
            if self.bbox.valid() and self.bbox.shape() != self._array.shape[0:2]:
                imbbox(self.clone().rgb()._array, self.bbox.xmin,
                       self.bbox.ymin, self.bbox.xmax, self.bbox.ymax,
                       bboxcaption=self.category(), colormap=colormap,
                       figure=figure, do_updateplot=flip)
            else:
                super(ImageDetection, self).show(figure=figure,
                                                 colormap=colormap, flip=flip)
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
            self.bbox = BoundingBox(xmin=float(xmin), ymin=float(ymin),
                                    xmax=float(xmax), ymax=float(ymax))
        elif bbox is not None:
            self.bbox = bbox
        elif (xmin is not None and ymin is not None and
              width is not None and height is not None):
            try:
                self.bbox = BoundingBox(xmin=float(xmin), ymin=float(ymin),
                                        width=float(width),
                                        height=float(height))
            except ValueError as e:
                print(('{}: xmin={}, ymin={}, width={}, height={}'.format(
                    e, xmin, ymin, width, height)))
                raise e
        elif (xcentroid is not None and ycentroid is not None and
              width is not None and height is not None):
            self.bbox = BoundingBox(xcentroid=float(xcentroid),
                                    ycentroid=float(ycentroid),
                                    width=float(width),
                                    height=float(height))
        elif (dilate is None and dilate_topheight is None):
            raise ValueError('Invalid bounding box - Either rect coordinates '
                             'or bbox object must be provided')

        if dilate_topheight is not None:
            self.bbox.dilate_topheight(dilate_topheight)
        if dilate is not None:
            self.bbox.dilate(dilate)
        return self

    def drawbox(self, bbox=None, border=None, color=None, alpha=None,
                beta=None):
        bbox = bbox if bbox is not None else self.boundingbox()
        super(ImageDetection, self).drawbox(bbox=bbox, border=border,
                                            color=color,
                                            alpha=alpha, beta=beta)
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
        return (self.bbox.xmin >= 0 and self.bbox.ymin >= 0 and
                self.bbox.xmax < W and self.bbox.ymax < H)

    def rescale(self, scale=1):
        """Rescale image buffer and bounding box - Not idemponent"""
        # image class inheritance
        self = super(ImageDetection, self).rescale(scale)
        self.bbox = self.bbox.rescale(scale)
        return self


    def resize(self, cols=None, rows=None):
        """Resize image buffer and bounding box"""
        # image class inheritance
        self = super(ImageDetection, self).resize(cols, rows)
        self.bbox = BoundingBox(xmin=0, ymin=0,
                                xmax=self.width(), ymax=self.height())
        return self

    def fliplr(self):
        """Mirror buffer and bounding box around vertical axis"""
        self = super(ImageDetection, self).fliplr()  # image class inheritance
        xmin = self.bbox.xmin
        xmax = self.bbox.xmax
        self.bbox.xmin = self.width() - xmax
        self.bbox.xmax = self.bbox.xmin + (xmax-xmin)
        return self

    def crop(self, bbox=None):
        """Crop image and update bounding box"""
        # image class inheritance
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

    def clip(self, border=0):
        self.bbox = self.bbox.intersection(
            BoundingBox(
                xmin=border,
                ymin=border,
                xmax=self.width()-border,
                ymax=self.height()-border))
        return self

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
        (ymin, ymax, xmin, xmax) = (int(np.round(self.bbox.ymin)),
                                    int(np.round(self.bbox.ymax)),
                                    int(np.round(self.bbox.xmin)),
                                    int(np.round(self.bbox.xmax)))
        immask[ymin:ymax, xmin:xmax] = 1
        return immask

    def setzero(self, bbox=None):
        """Set all image values within the bounding box to zero"""
        bbox = self.bbox if bbox is None else bbox
        self.load()[int(bbox.ymin):int(bbox.ymax),
                    int(bbox.xmin):int(bbox.xmax)] = 0
        return self
