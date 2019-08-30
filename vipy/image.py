import os
# import csv
from vipy.show import imshow, imbbox
from vipy.util import isnumpy, quietprint, isurl, islist, \
    fileext, tempimage, mat2gray, imwrite, imwritejet, imwritegray
# from strpy.bobo.util import isstring, tempcsv, istuple, remkdir, filetail
from vipy.geometry import BoundingBox, similarity_imtransform, \
    similarity_imtransform2D, imtransform, imtransform2D
from vipy import viset
import urllib.request
import urllib.error
import urllib.parse
import http.client as httplib
import cv2
import copy
import numpy as np
# import strpy.bobo.viset.download
import shutil

# FIX <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
# verify failed (_ssl.c:581)>
# http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


class Image(object):
    """Standard class for images"""
    def __init__(self, filename=None, url=None, ignore=False,
                 fetch=True, attributes=None):
        # # Private attributes
        # ignore errors during load? (useful for skipping broken links
        # if download fails)
        self._ignoreErrors = ignore
        # fetch url if not valid filename? (useful for skipping broken
        # links without download timeout)
        self._doFetch = fetch
        self._isdirty = False    # Locally modified image data contents?
        self._urluser = None      # basic authentication
        self._urlpassword = None  # basic authentication
        self._urlsha1 = None
        self._filename = None
        self._url = None
        self._loader = None

        # Initialization (filename or URL)
        if filename is not None:
            # image filename on local filesystem
            self._filename = str(filename)
            self._url = url
        if filename is not None and isurl(filename):
            self._url = str(filename)  # URL for image
            self._filename = None
        if url is not None and os.path.isfile(str(url)):
            self._filename = str(url)  # URL for image
            self._url = None
        if url is not None:
            self._url = str(url)  # URL for image
            # filename to be downloaded to (optional)
            self._filename = filename

        # Public attributes
        self.data = None      # Loaded image data
        self.attributes = attributes  # useful image attributes

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Yield single image for consistency with videos and templates"""
        yield self

    def __len__(self):
        """Images have length 1 always"""
        return 1

    def __repr__(self):
        # str_dirty = ", dirty" if self._isdirty else ""
        if self.isloaded():
            str_size = ", height=%d, width=%d, color='%s'" % (
                self.data.shape[0], self.data.shape[1],
                str(self.getattribute('colorspace')))
        else:
            str_size = ""
        str_file = "filename='%s'" % str(self._filename)
        if self._url is None:
            str_url = ''
        else:
            str_url = "url='%s', " % str(self._url)
        return str('<strpy.image: %s%s%s>' % (str_url, str_file, str_size))

    def tonumpy(self, ignoreErrors=False, asRaw=False, fetch=True):
        """load image, and return a numpy array and flush the underlying
        image."""
        img = self.load(ignoreErrors=ignoreErrors, asRaw=asRaw, fetch=fetch)
        self.flush()
        return img

    def loader(self, f):
        self._loader = f
        return self

    def load(self, ignoreErrors=False, asRaw=False, fetch=True):
        """Load image and return numpy array"""
        try:
            # Return previously loaded image
            if self.data is not None:
                return self.data

            # Download file
            if self._filename is not None and os.path.isfile(self._filename):
                quietprint('[strpy.image]: loading "%s" ' %
                           self._filename, verbosity=3)
            elif self._url is not None and fetch is True:
                quietprint('[strpy.image]: loading "%s" ' %
                           self._url, verbosity=3)
                if self._doFetch is True:
                    self.download(ignoreErrors=ignoreErrors)
            else:
                raise IOError('[strpy.image][WARNING]: No image to load %s' %
                              str(self))

            # Load file to numpy array
            if fileext(self._filename) == '.npz':
                if self._loader is None:
                    raise ValueError('Must define a customer loader for '
                                     '.npz file format')
                self.data = self._loader(self._filename)
            else:
                # BGR color order!
                self.data = cv2.imread(self._filename,
                                       cv2.CV_LOAD_IMAGE_UNCHANGED) \
                                       if asRaw else cv2.imread(self._filename)
            if self.data is None:
                if fileext(self._filename) == '.gif':
                    quietprint('[strpy.image][WARNING]: IO error - could '
                               'not load "%s" using opencv, '
                               'falling back on PIL ' %
                               self._filename, 1)
                    import PIL
                    # Convert .gif to luminance (grayscale) and export
                    # as numpy array
                    self.data = np.array(
                        PIL.Image.open(self._filename).convert('L'))
                else:
                    raise ValueError('invalid image file "%s"' %
                                     self._filename)

            # Image Atributes
            self.setattribute('colorspace',
                              'bgr' if self.data.ndim == 3 else 'gray')

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[strpy.image][WARNING]: IO error - '
                           'Invalid image file, url or invalid write '
                           'permissions "%s" ' %
                           self.filename(), True)
                self.data = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[strpy.image][WARNING]: '
                           'load error for image "%s"' %
                           self.filename(), verbosity=2)
                self.data = None
            else:
                raise

        return self.data

    def fetch(self, ignoreErrors=False, asRaw=False):
        """Fetch image from filename and return image object"""
        self.load(ignoreErrors=ignoreErrors, asRaw=asRaw)
        return self

    def download(self, ignoreErrors=False, timeout=10):
        """Download URL to provided filename"""
        if self._url is None or not isurl(str(self._url)):
            raise ValueError('[strpy.image.download][ERROR]: '
                             'Invalid URL "%s" ' % self._url)
        if self._filename is None:
            self._filename = tempimage(fileext(self._url))
        try:
            url_scheme = urllib.parse.urlparse(self._url)[0]
            if url_scheme in ['http', 'https']:
                quietprint('[strpy.image.download]: '
                           'downloading "%s" to "%s" ' %
                           (self._url, self._filename), verbosity=1)
                viset.download.download(self._url,
                                        self._filename,
                                        verbose=False,
                                        timeout=timeout,
                                        sha1=self._urlsha1,
                                        username=self._urluser,
                                        password=self._urlpassword)
            elif url_scheme == 'file':
                quietprint('[strpy.image.download]: copying "%s" to "%s" ' %
                           (self._url, self._filename), verbosity=2)
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
                quietprint('[strpy.image][WARNING]: download failed - '
                           'ignoring image', 1)
                self.data = None
            else:
                raise

        except IOError:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[strpy.image][WARNING]: IO error - '
                           'Invalid image file, url or '
                           'invalid write permissions "%s" ' %
                           self.filename(), True)
                self.data = None
            else:
                raise

        except KeyboardInterrupt:
            raise

        except Exception:
            if self._ignoreErrors or ignoreErrors:
                quietprint('[strpy.image][WARNING]: '
                           'load error for image "%s"' %
                           self.filename(), verbosity=2)
            else:
                raise
        self.flush()
        return self

    def flush(self):
        self.data = None
        self._isdirty = False
        return self

    def reload(self):
        self.flush().load()
        return self

    def show(self, ignoreErrors=False, colormap=None, figure=None, flip=True):
        if self.load(ignoreErrors=ignoreErrors) is not None:
            quietprint('[strpy.image][%s]: displaying image' %
                       (self.__repr__()), verbosity=2)
            if self.iscolor():
                if colormap == 'gray':
                    imshow(self.clone().grayscale().rgb().data, figure=figure)
                else:
                    if colormap is not None:
                        quietprint('[strpy.image][%s]: '
                                   'ignoring colormap for color image' %
                                   (self.__repr__()), verbosity=2)

                    imshow(self.clone().rgb().data, figure=figure)
            else:
                imshow(self.load(), colormap=colormap,
                       figure=figure, do_updateplot=flip)
        return self

    def imagesc(self):
        if self.load(ignoreErrors=False) is not None:
            imshow(self.clone().mat2gray().rgb().data)
        return self

    def mediatype(self):
        return 'image'

    def iscolor(self):
        return self.load().ndim == 3

    def isvalid(self, flush=False):
        if self.filename() is not None:
            v = os.path.isfile(self.filename()) and self.load() is not None
            if flush:
                self.flush()
            return v
        else:
            return False

    def bytes(self):
        if not self.isvalid():
            self.download()
        return os.path.getsize(self._filename)

    def width(self):
        return self.load().shape[1]

    def height(self):
        return self.load().shape[0]

    def array(self, data=None):
        if data is None:
            return self.data
        elif isnumpy(data):
            self.data = np.copy(data)
            self._filename = None
            self._url = None
            return self
        else:
            raise ValueError('Invalid numpy array')

    def buffer(self, data=None):
        return self.array(data)

    def setfilename(self, newfile):
        # set filename and return object
        self.flush()
        self._filename = newfile
        return self

    def filename(self, newfile=None):
        """Image Filename"""
        if newfile is None:
            return self._filename
        else:
            # set filename and return object
            self.flush()
            self._filename = newfile
            return self

    def url(self, url=None, username=None, password=None, sha1=None):
        """Image URL"""
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
        return self

    def isloaded(self):
        return self.data is not None


#    def savein(self, dirname):
#        newfile = os.path.join(dirname, filetail(self.filename()))
#        remkdir(dirname)
#        quietprint('[strpy.image]: saving "%s"' % newfile)
#        if self.load().ndim == 3:
#            # if opened with PIL
#            # cv2.imwrite(newfile, cv2.cvtColor(self.load(),
#            #             cv2.COLOR_BGR2RGB))
#            cv2.imwrite(newfile, self.load())
#        else:
#            cv2.imwrite(newfile, self.load())
#        self.url = newfile
#        self._cachekey = newfile
#        self._isdirty = False
#        self.data = None  # force reload
#        return self

    def saveas(self, filename):
        if self.load().ndim == 3:
            self.imwrite(filename)
        else:
            self.imwritegray(filename)
        self.flush()
        self._filename = filename
        return self

    def imwritegray(self, filename):
        return imwritegray(self.grayscale().data, filename)

    def imwritejet(self, filename):
        return imwritejet(self.grayscale().float().data, filename)

    def imwrite(self, filename, writeas=None):
        return imwrite(self.load(), filename, writeas=writeas)

#    def writeback(self):
#        im = self.clone()
#        if im._isdirty:
#            ext = fileext(im.filename())
#            # force .gif extension to be .jpg for imwrite
#            ext = ext if ext != '.gif' else '.jpg'
#            im = im.saveas(tempimage(ext=ext))
#        return im

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

    # MODIFY IMAGE ---------------------------------------------------------
    def clone(self):
        """Create deep copy of image object"""
        im = copy.deepcopy(self)
        if self.data is not None:
            im.data = self.data.copy()
        return im

    def resize(self, cols=None, rows=None):
        """Resize the image buffer to (rows x cols)"""
        if cols is None or rows is None:
            if cols is None:
                scale = float(rows)/float(self.height())
            else:
                scale = float(cols)/float(self.width())
            quietprint('[strpy.image][%s]: scale=%1.2f' %
                       (self.__repr__(), scale), verbosity=2)
            # OpenCV decimation introduces artifacts using cubic
            # interp, INTER_AREA is recommended according to the
            # OpenCV docs
            interp_method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            self.data = cv2.resize(self.load(), dsize=(0, 0),
                                   fx=scale, fy=scale,
                                   interpolation=interp_method)
        else:
            quietprint('[strpy.image][%s]: resize=(%d,%d)' %
                       (self.__repr__(), rows, cols), verbosity=2)
            try:
                interp_method = cv2.INTER_AREA if (
                    rows < self.height() or
                    cols < self.width()) else cv2.INTER_CUBIC
                # fixed bug since opencv takes x and y not rows, cols
                self.data = cv2.resize(self.load(), dsize=(cols, rows),
                                       interpolation=interp_method)
            except:
                print(self)  # DEBUGGING
                raise

        dtype = self.data.dtype
        if dtype == np.float32 or dtype == np.float64:
            np.clip(self.data, 0.0, 1.0, out=self.data)
        self._isdirty = True
        return self

    def rescale(self, scale=1):
        """Scale the image buffer by the given factor - NOT idemponent"""
        quietprint('[strpy.image][%s]: scale=%1.2f to (%d,%d)' %
                   (self.__repr__(), scale, scale*self.width(),
                    scale*self.height()), verbosity=2)
        # OpenCV decimation introduces artifacts using cubic interp ,
        # INTER_AREA is recommended according to the OpenCV docs
        interp_method = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        self.data = cv2.resize(self.load(), dsize=(0, 0),
                               fx=scale, fy=scale, interpolation=interp_method)
        dtype = self.data.dtype
        if dtype == np.float32 or dtype == np.float64:
            np.clip(self.data, 0.0, 1.0, out=self.data)
        self._isdirty = True
        return self

    def pad(self, dx, dy, mode='edge'):
        """Pad image using np.pad mode"""
        self.data = np.pad(self.load(),
                           ((dx, dx), (dy, dy), (0, 0)) if
                           self.load().ndim == 3 else ((dx, dx), (dy, dy)),
                           mode=mode)
        self._isdirty = True
        return self

    def zeropad(self, dx, dy):
        """Pad image using np.pad constant"""
        if self.load().ndim == 3:
            pad_width = ((dx, dx), (dy, dy), (0, 0))
            constant_values = (0, 0, 0)
        else:
            pad_width = ((dx, dx), (dy, dy))
            constant_values = (0, 0)
        self.data = np.pad(self.load(),
                           pad_width=pad_width,
                           mode='constant',
                           constant_values=constant_values)
        self._isdirty = True
        return self

    def meanpad(self, dx, dy):
        """Pad image using np.pad constant where constant is image mean"""
        mu = self.mean()
        if self.load().ndim == 3:
            pad_size = ((dx, dx), (dy, dy), (0, 0))
            constant_values = tuple([(x, y) for (x, y) in zip(mu, mu)])
        else:
            constant_values = ((mu, mu), (mu, mu))
            pad_size = ((dx, dx), (dy, dy))
        self.data = np.pad(self.load(), pad_size, mode='constant',
                           constant_values=constant_values)
        self._isdirty = True
        return self

    def crop(self, bbox=None, pad='mean'):
        """Crop the image buffer using the supplied bounding box - NOT
        idemponent."""
        if bbox is not None:
            if islist(bbox):
                bbox = BoundingBox(xmin=bbox[0], ymin=bbox[1],
                                   xmax=bbox[2], ymax=bbox[3])

            bbox = bbox.imclip(self.load())  # FIXME
            quietprint('[strpy.image][%s]: cropping "%s"' %
                       (self.__repr__(), str(bbox)), verbosity=2)
            bbox = bbox.imclip(self.load())
            # assumed numpy
            self.data = self.load()[int(bbox.ymin):int(bbox.ymax),
                                    int(bbox.xmin):int(bbox.xmax)]
            self._isdirty = True
        return self

    def fliplr(self):
        """Mirror the image buffer about the vertical axis - Not idemponent"""
        quietprint('[strpy.image][%s]: fliplr' %
                   (self.__repr__()), verbosity=2)
        self.data = np.fliplr(self.load())
        self._isdirty = True
        return self

    def raw(self, normalized=True):
        """Load the image as a raw image buffer"""
        quietprint('[strpy.image][%s]: loading raw imagery data' %
                   (self.__repr__()), verbosity=2)
        self.data = self.load(asRaw=True)
        return self

    def grayscale(self):
        """Convert the image buffer to grayscale"""
        if self.load().ndim == 3:
            quietprint('[strpy.image][%s]: converting to grayscale' %
                       (self.__repr__()), verbosity=3)
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_BGR2GRAY)
            self._isdirty = True
            self.setattribute('colorspace', 'gray')
        return self

    def rgb(self):
        """Convert the image buffer to RGB"""
        if self.load().ndim == 3:
            quietprint('[strpy.image][%s]: converting bgr to rgb' %
                       (self.__repr__()), verbosity=2)
            # opencv BGR to RGB
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_BGR2RGB)
        elif self.load().ndim == 2:
            quietprint('[strpy.image][%s]: converting gray to rgb' %
                       (self.__repr__()), verbosity=2)
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_GRAY2RGB)
        self._isdirty = True
        self.setattribute('colorspace', 'rgb')
        return self

    def hsv(self):
        """Convert the image buffer to HSV color space"""
        if self.iscolor():
            quietprint('[strpy.image][%s]: converting to hsv' %
                       (self.__repr__()), verbosity=2)
            # opencv BGR (assumed) to HSV
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_BGR2HSV)
            self._isdirty = True
            self.setattribute('colorspace', 'hsv')
        else:
            quietprint('[strpy.image][%s]: converting grayscale to hsv' %
                       (self.__repr__()), verbosity=2)
            # grayscale -> RGB -> HSV (HACK)
            self.data = cv2.cvtColor(self.rgb().load(), cv2.COLOR_RGB2HSV)
        return self

    def bgr(self):
        """Convert the image buffer to BGR color format"""
        if self.load().ndim == 3:
            # Loaded by opencv, so assumed BGR order
            self.data = self.load()
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_RGB2BGR)
        elif self.load().ndim == 2:
            quietprint('[strpy.image][%s]: converting gray to bgr' %
                       (self.__repr__()), verbosity=2)
            self.data = cv2.cvtColor(self.load(), cv2.COLOR_GRAY2BGR)
        self.setattribute('colorspace', 'bgr')
        self._isdirty = True
        return self

    def float(self, scale=None):
        """Convert the image buffer to float32"""
        if self.load().dtype != np.float32:
            quietprint('[strpy.image][%s]: converting to float32' %
                       (self.__repr__()), verbosity=2)
            self.data = np.float32(self.load())
        if scale is not None:
            self.data = self.data * scale
        return self

    def uint8(self, scale=None):
        """Convert the image buffer to uint8"""
        if scale is not None:
            self.data = self.load() * scale
        if self.load().dtype != np.uint8:
            quietprint('[strpy.image][%s]: converting to uint8' %
                       (self.__repr__()), verbosity=2)
            self.data = np.uint8(self.load())
        return self

    def preprocess(self, scale=1.0/255.0):
        """Preprocess the image buffer by converting to grayscale, convert to
        float, then rescaling [0,255], [0,1] - Not idemponent."""
        quietprint('[strpy.image][%s]: preprocessing' %
                   (self.__repr__()), verbosity=2)
        self = self.grayscale().float()
        self.data = scale*self.data
        return self

    def min(self):
        return np.min(self.load().flatten())

    def max(self):
        return np.max(self.load().flatten())

    def mean(self):
        return np.mean(self.load(), axis=(0, 1)).flatten()

    def mat2gray(self, min=None, max=None):
        """Convert the image buffer so that [min,max] -> [0,1]"""
        quietprint('[strpy.image][%s]: contrast equalization' %
                   (self.__repr__()), verbosity=2)
        self.data = mat2gray(np.float32(self.load()), min, max)
        self._isdirty = True
        return self

    def transform2D(self, txy=(0, 0), r=0, s=1):
        """Transform the image buffer using a 2D similarity transform - Not
        idemponent."""
        quietprint('[strpy.image][%s]: transform2D' %
                   (self.__repr__()), verbosity=2)
        self.load()
        c = (self.data.shape[1] / 2, self.data.shape[0] / 2)
        M = similarity_imtransform2D(c=c, r=r, s=s)
        self.data = imtransform2D(self.data, M)
        A = similarity_imtransform(txy=txy)
        self.data = imtransform(self.data, A)
        return self

    def transform(self, A):
        """Transform the image buffer using the supplied affine transformation
        - Not idemponent."""
        quietprint('[strpy.image][%s]: transform' %
                   (self.__repr__()), verbosity=2)
        self.data = imtransform(self.load(), A)
        return self

    def map(self, f):
        """Transform the image object using the supplied function - May not be
        idemponent."""
        return f(self)

    def drawbox(self, bbox, border=None, color=None, alpha=None, beta=None):
        self.load()
        dtype = self.data.dtype

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

        data = self.data.astype(np.float32)
        data[ymin:ymax, xmin:xmax] *= alpha
        data[ymin:ymax, xmin:xmax] += beta

        data = np.clip(data, 0, clip)
        data = data.astype(dtype)

        self.data[ymin:ymin+border, xmin:xmax] = color
        self.data[ymax-border:ymax, xmin:xmax] = color
        self.data[ymin:ymax, xmin:xmin+border] = color
        self.data[ymin:ymax, xmax-border:xmax] = color

        return self

    def html(self, alt=None):
        im = self.clone().rgb()
        ret, data = cv2.imencode('.png', im.data)
        b = data.tobytes().encode('base64')
        alt_text = alt if alt is not None else im.filename()
        return '<img src="data:image/png;base64,%s" alt="%s" />' % (b,
                                                                    alt_text)


class ImageCategory(Image):
    def __init__(self, filename=None, url=None, category=None,
                 ignore=False, fetch=True, attributes=None):
        # Image class inheritance
        super(ImageCategory, self).__init__(filename=filename,
                                            url=url,
                                            ignore=ignore,
                                            fetch=fetch,
                                            attributes=attributes)
        self._category = category

    def __repr__(self):
        # str_dirty = ", dirty" if self._isdirty else ""
        if self.isloaded():
            str_size = ", height=%d, width=%d, color='%s'" % (
                self.data.shape[0], self.data.shape[1],
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
        return str('<strpy.imagecategory: %s%s%s>' % (str_file,
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
                 ignore=False, fetch=True, attributes=None, xmin=float('nan'),
                 xmax=float('nan'), ymin=float('nan'), ymax=float('nan'),
                 bbox=None):
        # ImageCategory class inheritance
        super(ImageDetection, self).__init__(filename=filename,
                                             url=url,
                                             ignore=ignore,
                                             fetch=fetch,
                                             attributes=attributes,
                                             category=category)

        if bbox is not None:
            self.bbox = bbox
        else:
            self.bbox = BoundingBox(xmin=float(xmin), ymin=float(ymin),
                                    xmax=float(xmax), ymax=float(ymax))

    def __repr__(self):
        # str_dirty = ", dirty" if self._isdirty else ""
        if self.isloaded():
            str_size = ", height=%d, width=%d, color='%s'" % (
                self.data.shape[0],
                self.data.shape[1],
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

        return str('<strpy.imagedetection: %s%s%s%s>' % (
            str_category, str_detection, str_file, str_size))

    def __hash__(self):
        return hash(self.__repr__())

    def show(self, ignoreErrors=False, colormap=None, figure=None, flip=True):
        if self.load(ignoreErrors=ignoreErrors) is not None:
            if self.bbox.valid() and self.bbox.shape() != self.data.shape[0:2]:
                imbbox(self.clone().rgb().data, self.bbox.xmin,
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
