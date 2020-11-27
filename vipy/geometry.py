import math
import numpy as np
from itertools import product
from vipy.util import try_import, istuple, isnumpy, isnumber, tolist
from vipy.linalg import columnvector
import warnings

try:
    import ujson as json  # faster
except ImportError:
    import json


def covariance_to_ellipse(cov):
    """2x2 covariance matrix to ellipse (major_axis_length, minor_axis_length, angle_in_radians)"""
    assert isnumpy(cov) and cov.shape == (2,2), "Invalid input"
    (d,V) = np.linalg.eig(cov)
    return np.array((d[0], d[1], math.atan2(V[1,0], V[0,0])))  # (major_axis_len, minor_axis_len, angle_in_radians)


def dehomogenize(p):
    """Convert 3x1 homogenous point (x,y,h) to 2x1 non-homogenous point (x/h, y/h)"""
    assert isnumpy(p)    
    if p.ndim == 1:
        assert len(p) == 3
        return p[0:2] / p[2]
    elif p.ndim == 2:
        assert isnumpy(p) and p.shape[0] == 3, "Invalid input"
        p = columnvector(p) if p.ndim == 1 else p
        return p[0:-1, :] / p[-1,:]
    else:
        return ValueError('p must be 1d or 2d')
    

def homogenize(p):
    """Convert 2xN non-homogenous points (x,y) to 3xN non-homogenous point (x, y, 1)"""
    assert isnumpy(p)
    if p.ndim == 1:
        return np.hstack( (p, 1) )
    elif p.ndim == 2:
        assert p.shape[0] == 2, "Invalid input"
        p = columnvector(p) if p.ndim == 1 else p
        return np.vstack((p, np.ones_like(p[-1])))
    else:
        return ValueError('p must be 1d or 2d')


def apply_homography(H,p):
    """Apply a 3x3 homography H to non-homogenous point p and return a transformed point """
    assert isnumpy(H) and isnumpy(p) and H.shape == (3,3) and p.shape[0] == 2, "Invalid input"
    return dehomogenize(np.dot(H, homogenize(p)))


def similarity_transform_2x3(c=(0,0), r=0, s=1):
    """Return a 2x3 similarity transform with rotation r (radians), scale s and origin c=(x,y)"""
    assert istuple(c) and len(c) == 2 and isnumber(r) and isnumber(s), "Invalid input"
    deg = r * 180. / math.pi
    a = s * np.cos(r)
    b = s * np.sin(r)
    (x,y) = (c[0], c[1])
    return np.array([[a, b, (1 - a) * x - b * y], [-b, a, b * x + (1 - a) * y]])


def similarity_transform(txy=(0,0), r=0, s=1):
    """Return a 3x3 similarity transformation with translation tuple txy=(x,y), rotation r (radians, scale=s"""
    assert istuple(txy) and len(txy) == 2 and isnumber(r) and isnumber(s), "Invalid input"
    R = np.mat([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0,0,1]])
    S = np.mat([[s,0,0], [0, s, 0], [0,0,1]])
    T = np.mat([[0,0,txy[0]], [0,0,txy[1]], [0,0,0]])
    return S * R + T  # composition


def affine_transform(txy=(0,0), r=0, sx=1, sy=1, kx=0, ky=0):
    """Compose and return a 3x3 affine transformation for translation txy=(0,0), rotation r (radians), scalex=sx, scaley=sy, shearx=kx, sheary=ky"""
    assert istuple(txy) and len(txy) == 2 and isnumber(r) and isnumber(sx) and isnumber(sy) and isnumber(kx) and isnumber(ky), "Invalid input"
    R = np.mat([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0,0,1]])
    S = np.mat([[sx,0,0], [0, sy, 0], [0,0,1]])
    K = np.mat([[1,ky,0], [kx,1,0], [0,0,1]])
    T = np.mat([[0,0,txy[0]], [0,0,txy[1]], [0,0,0]])
    return K * S * R + T  # composition


def random_affine_transform(txy=((0,1),(0,1)), r=(0,1), sx=(0.1,1), sy=(0.1,1), kx=(0.1,1), ky=(0.1,1)):
    """Return a random 3x3 affine transformation matrix for the provided ranges, inputs must be tuples"""
    assert istuple(txy) and istuple(txy[0]) and istuple(txy[1]) and istuple(r) and istuple(sx) and istuple(sy) and istuple(kx) and istuple(ky), "Invalid input"
    uniform_random_in_range = lambda t: np.random.uniform(t[0], t[1])
    return affine_transform(txy=(uniform_random_in_range(txy[0]), uniform_random_in_range(txy[1])),
                            r=uniform_random_in_range(r),
                            sx=uniform_random_in_range(sx),
                            sy=uniform_random_in_range(sy),
                            kx=uniform_random_in_range(kx),
                            ky=uniform_random_in_range(ky))


def imtransform(img, A):
    """Transform an numpy array image (MxNx3) following the affine or similiarity transformation A"""
    assert isnumpy(img) and isnumpy(A), "invalid input"
    try_import(cv2, 'opencv-python'); import cv2
    if A.shape == (2,3):
        return cv2.warpAffine(img, A, (img.shape[1], img.shape[0]))
    else:
        return cv2.warpPerspective(img, A, (img.shape[1], img.shape[0]))


def normalize(x, eps=1E-16):
    """Given a vector x, return the vector unit normalized as float64"""
    assert isnumpy(x), "Invalid input"
    return x / (np.linalg.norm(x.astype(np.float64)) + eps)

def imagebox(shape):
    return BoundingBox(xmin=0, ymin=0, width=shape[1], height=shape[0])



class BoundingBox(object):
    """Core bounding box class with flexible constructors in this priority order:
          (xmin,ymin,xmax,ymax)
          (xmin,ymin,width,height)
          (centroid[0],centroid[1],width,height)
          (xcentroid,ycentroid,width,height)
          xywh=(xmin,ymin,width,height)
          ulbr=(xmin,ymin,xmax,ymax)
          bounding rectangle of binary mask image"""

    #__slots__ = ['_xmin', '_ymin', '_xmax', '_ymax']  # This is not backwards compatible
    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, centroid=None, xcentroid=None, ycentroid=None, width=None, height=None, mask=None, xywh=None, ulbr=None, ulbrdict=None):

        if ulbrdict is not None:
            self.__dict__ = ulbrdict  # equivalent to (but faster)
            #self._xmin = ulbrdict['_xmin']
            #self._ymin = ulbrdict['_ymin']
            #self._xmax = ulbrdict['_xmax']
            #self._ymax = ulbrdict['_ymax']                                  
        elif xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            if not (isnumber(xmin) and isnumber(ymin) and isnumber(xmax) and isnumber(ymax)):
                raise ValueError('Box coordinates must be integers or floats not "%s"' % str(type(xmin)))
            self._xmin = float(xmin)
            self._ymin = float(ymin)
            self._xmax = float(xmax)
            self._ymax = float(ymax)
        elif xmin is not None and ymin is not None and width is not None and height is not None:
            if not (isnumber(xmin) and isnumber(ymin) and isnumber(width) and isnumber(height)):
                raise ValueError('Box coordinates must be integers or floats not "%s"' % str(type(width)))
            self._xmin = float(xmin)
            self._ymin = float(ymin)
            self._xmax = self._xmin + float(width)
            self._ymax = self._ymin + float(height)
        elif centroid is not None and width is not None and height is not None:
            if not (len(centroid) == 2 and isnumber(centroid[0]) and isnumber(centroid[1]) and isnumber(width) and isnumber(height)):
                raise ValueError('Invalid box coordinates')
            self._xmin = float(centroid[0]) - float(width) / 2.0
            self._ymin = float(centroid[1]) - float(height) / 2.0
            self._xmax = float(centroid[0]) + float(width) / 2.0
            self._ymax = float(centroid[1]) + float(height) / 2.0
        elif xcentroid is not None and ycentroid is not None and width is not None and height is not None:
            #if not (isnumber(xcentroid) and isnumber(ycentroid) and isnumber(width) and isnumber(height)):
            #    raise ValueError('Box coordinates must be integers or floats')
            self._xmin = float(xcentroid) - (float(width) / 2.0)
            self._ymin = float(ycentroid) - (float(height) / 2.0)
            self._xmax = float(xcentroid) + (float(width) / 2.0)
            self._ymax = float(ycentroid) + (float(height) / 2.0)
        elif xywh is not None:
            self.xywh(xywh)
        elif ulbr is not None:
            self.ulbr(ulbr)
        elif mask is not None:
            # Bounding rectangle of non-zero pixels in a binary mask image
            if not isnumpy(mask) or np.sum(mask) == 0:
                raise ValueError('Mask input must be numpy array with at least one non-zero entry')
            imx = np.sum(mask, axis=0)
            imy = np.sum(mask, axis=1)
            self._xmin = np.argwhere(imx > 0)[0]
            self._ymin = np.argwhere(imy > 0)[0]
            self._xmax = np.argwhere(imx > 0)[-1]
            self._ymax = np.argwhere(imy > 0)[-1]
        else:
            raise ValueError('invalid constructor input')

    @classmethod
    def cast(cls, bb, flush=False):
        assert isinstance(bb, BoundingBox)
        bb.__class__ = BoundingBox
        if flush:
            bb.__dict__ = {k:v for (k,v) in bb.__dict__.items() if k in ['_xmin', '_ymin', '_xmax', '_ymax']}        
        return bb
    
    @classmethod
    def from_json(cls, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        return cls(ulbrdict=d)

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(encode=False)
    
    def json(self, encode=True):
        return json.dumps(self.__dict__) if encode else self.__dict__
        
    def clone(self):
        return BoundingBox(xmin=self._xmin, xmax=self._xmax, ymin=self._ymin, ymax=self._ymax)
    def bbclone(self):
        return BoundingBox(xmin=self._xmin, xmax=self._xmax, ymin=self._ymin, ymax=self._ymax)

    def __eq__(self, other):
        """Bounding box equality"""
        return isinstance(other, BoundingBox) and self.xywh() == other.xywh()

    def __neq__(self, other):
        """Bounding box non-equality"""
        return not self.__eq__(other)

    def __repr__(self):
        return str('<vipy.geometry.boundingbox: xmin=%s, ymin=%s, width=%s, height=%s>' % (self.xmin(), self.ymin(), self.bbwidth(), self.bbheight()))

    def __str__(self):
        return self.__repr__()

    def xmin(self):
        """x coordinate of upper left corner of box, x-axis is image column"""
        return self._xmin

    def ymin(self):
        """y coordinate of upper left corner of box, y-axis is image row"""
        return self._ymin

    def xmax(self):
        """x coordinate of lower right corner of box, x-axis is image column"""
        return self._xmax

    def ymax(self):
        """y coordinate of lower right corner of box, y-axis is image row"""
        return self._ymax

    def upperleft(self):
        """Return the (x,y) upper left corner coordinate of the box"""
        return (self.xmin(), self.ymin())

    def bottomleft(self):
        """Return the (x,y) lower left corner coordinate of the box"""
        return (self.xmin(), self.ymax())

    def upperright(self):
        """Return the (x,y) upper right corner coordinate of the box"""
        return (self.xmax(), self.ymin())

    def bottomright(self):
        """Return the (x,y) lower right corner coordinate of the box"""
        return (self.xmax(), self.ymax())

    def int(self):
        """Convert corners to integer with rounding"""
        (w,h) = (int(np.round(self.bbwidth())), int(np.round(self.bbheight())))
        self._xmin = int(np.round(self._xmin))
        self._ymin = int(np.round(self._ymin))
        self._xmax = int(np.round(self._xmax))
        self._ymax = int(np.round(self._ymax))
        if w != self.bbwidth():
            self.right(w - self.bbwidth())  # preserve aspect ratio due to rounding by +/- right side of box 
        if h != self.bbheight():
            self.bottom(h-self.bbheight())  # preserve aspect ratio due to rounding by +/- bottom of box
        return self

    def float(self):
        """Convert corners to float"""
        self._xmin = float(self._xmin)
        self._ymin = float(self._ymin)
        self._xmax = float(self._xmax)
        self._ymax = float(self._ymax)
        return self

    def significant_digits(self, n):
        """Convert corners to have at most n significant digits for efficient JSON storage"""
        assert isinstance(n, int) and n>=0
        self._xmin = round(self._xmin, n)
        self._ymin = round(self._ymin, n)
        self._xmax = round(self._xmax, n)
        self._ymax = round(self._ymax, n)
        return self
        
    def translate(self, dx=0, dy=0):
        """Translate the bounding box by dx in x and dy in y"""
        self._xmin = self._xmin + dx
        self._ymin = self._ymin + dy
        self._xmax = self._xmax + dx
        self._ymax = self._ymax + dy
        return self

    def offset(self, dx=0, dy=0):
        """Alias for translate"""
        return self.translate(dx, dy)

    def invalid(self):
        """Is the box a valid bounding box?"""
        #is_undefined = np.isnan(self._xmin) or np.isnan(self._ymin) or np.isnan(self._xmax) or np.isnan(self._ymax)
        is_valid = ((self._xmax - self._xmin) >= 0) and ((self._ymax - self._ymin) >= 0)  # if nan, will return False
        return not is_valid

    def valid(self):
        return not self.invalid()

    def isvalid(self):
        return not self.invalid()

    def isdegenerate(self):
        return self.invalid()
        
    def isnonnegative(self):
        return (self.xmin() >= 0 and
                self.ymin() >= 0 and
                self.xmax() >= 0 and
                self.ymax() >= 0)

    def width(self):
        return self._xmax - self._xmin
    def bbwidth(self):
        return self._xmax - self._xmin
    
    def setwidth(self, w):
        """Set new width keeping centroid constant"""
        if w <= 0:
            raise ValueError('invalid width')
        worig = (self._xmax - self._xmin)
        self._xmax += float((w - worig) / 2.0)
        self._xmin -= float((w - worig) / 2.0)
        return self

    def setheight(self, h):
        """Set new height keeping centroid constant"""
        if h <= 0:
            raise ValueError('invalid height')
        horig = self._ymax - self._ymin
        self._ymax += float((h - horig) / 2.0)
        self._ymin -= float((h - horig) / 2.0)
        return self

    def height(self):
        return self._ymax - self._ymin
    def bbheight(self):
        return self._ymax - self._ymin

    def centroid(self, c=None):
        """(x,y) tuple of centroid position of bounding box"""
        if c is None:
            (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)            
            return (self._xmin + (float(width) / 2.0), self._ymin + (float(height) / 2.0))
        else:
            assert len(c) == 2
            (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)
            self._xmin = float(c[0]) - (width / 2.0)
            self._ymin = float(c[1]) - (height / 2.0)
            self._xmax = float(c[0]) + (width / 2.0)
            self._ymax = float(c[1]) + (height / 2.0)
            return self
            
    def x_centroid(self):
        return self.centroid()[0]

    def xcentroid(self):
        """Alias for x_centroid()"""
        return self.centroid()[0]
    def centroid_x(self):
        """Alias for x_centroid()"""
        return self.centroid()[0]
            
    def y_centroid(self):
        return self.centroid()[1]

    def ycentroid(self):
        """Alias for y_centroid()"""
        return self.centroid()[1]
    def centroid_y(self):
        """Alias for y_centroid()"""
        return self.centroid()[1]
    
    def area(self):
        """Return the area=width*height of the bounding box"""
        (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)        
        return width * height

    def to_xywh(self, xywh=None):
        """Return bounding box corners as (x,y,width,height) tuple"""
        if xywh is None:
            (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)                    
            return tuple([self._xmin, self._ymin, width, height])
        else:
            assert len(xywh) == 4, "Invalid (xmin,ymin,width,height) input"
            self._xmin = float(xywh[0])
            self._ymin = float(xywh[1])
            self._xmax = float(self._xmin + xywh[2])
            self._ymax = float(self._ymin + xywh[3])
            return self

    def xywh(self, xywh_=None):
        """Alias for to_xywh"""
        return self.to_xywh(xywh_)

    def cxywh(self, cxywh=None):
        """Return or set bounding box corners as (centroidx,centroidy,width,height) tuple"""
        if cxywh is None:
            (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)                    
            return tuple([self.x_centroid(), self.y_centroid(), width, height])
        else:
            assert len(cxywh) == 4, "Invalid (xcentroid, ycentroid, width, height) input"
            return self.centroid( (cxywh[0], cxywh[1]) ).setwidth(cxywh[2]).setheight(cxywh[3])            
    
    def ulbr(self, ulbr=None):
        """Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)"""
        if ulbr is None:
            return (self.xmin(), self.ymin(), self.xmax(), self.ymax())            
        else:
            assert len(ulbr) == 4, "Invalid (xmin,ymin,xmax,ymax) input"
            self._xmin = float(ulbr[0])
            self._ymin = float(ulbr[1])
            self._xmax = float(ulbr[2])
            self._ymax = float(ulbr[3])
            return self

    def to_ulbr(self, ulbr=None):
        """Alias for ulbr()"""
        return self.ulbr(ulbr)
    
    def dx(self, bb):
        """Offset bounding box by same xmin as provided box"""
        return bb._xmin - self._xmin

    def dy(self, bb):
        """Offset bounding box by ymin of provided box"""
        return bb._ymin - self._ymin

    def sqdist(self, bb):
        """Squared Euclidean distance between upper left corners of two bounding boxes"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))                
        return np.power(self.dx(bb), 2.0) + np.power(self.dy(bb), 2.0)

    def dist(self, bb):
        """Distance between centroids of two bounding boxes"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))                
        return np.sqrt(np.sum(np.square(np.array(bb.centroid()) - np.array(self.centroid()))))

    def pdist(self, bb, sigma=None):
        """Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))
        return np.exp(-self.sqdist(bb)/(float(2*self.maxdim()*self.maxdim()) if sigma is None else float(2.0*sigma*sigma)))

    def iou(self, bb):
        """area of intersection / area of union"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))        
        if bb is None or bb.invalid() or self.invalid():
            return 0.0
        w = min(self.xmax(), bb.xmax()) - max(self.xmin(), bb.xmin())
        h = min(self.ymax(), bb.ymax()) - max(self.ymin(), bb.ymin())
        if ((w < 0) or (h < 0)):
            iou = 0   # invalid (no overlap)
        else:
            area_intersection = w * h
            area_union = (self.area() + bb.area() - area_intersection)
            iou = area_intersection / float(area_union)
        return iou

    def intersection_over_union(self, bb):
        """Alias for iou"""
        return self.iou(bb)

    def area_of_intersection(self, bb):
        """area of intersection"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))                
        if bb.invalid() or self.invalid():
            return 0.0
        w = min(self.xmax(), bb.xmax()) - max(self.xmin(), bb.xmin())
        h = min(self.ymax(), bb.ymax()) - max(self.ymin(), bb.ymin())
        if ((w < 0) or (h < 0)):
            aoi = 0   # invalid (no overlap)
        else:
            aoi = w * h
        return aoi

    def cover(self, bb):
        """Fraction of this bounding box intersected by other bbox (bb)"""        
        return self.area_of_intersection(bb) / float(self.area())

    def shapeiou(self, bb):
        """Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation"""
        assert isinstance(bb, BoundingBox), "Invalid input - must be BoundingBox()"
        return self.iou(bb.clone().translate(dx=self.xmin()-bb.xmin(), dy=self.ymin()-bb.ymin()))

    def intersection(self, bb, strict=True):
        """Intersection of two bounding boxes, throw an error on degeneracy of intersection result (if strict=True)"""
        assert isinstance(bb, BoundingBox), "Invalid BoundingBox() input of type '%s'" % str(type(bb))                        
        self._xmin = max(bb.xmin(), self.xmin())
        self._ymin = max(bb.ymin(), self.ymin())
        self._xmax = min(bb.xmax(), self.xmax())
        self._ymax = min(bb.ymax(), self.ymax())
        if strict and self.isdegenerate():
            raise ValueError('Degenerate intersection for bounding boxes "%s" and "%s"' % (str(bb), str(self)))
        return self

    def hasintersection(self, bb):
        """Return true of self and bb intersect"""
        return self.area_of_intersection(bb) > 0

    def union(self, bb):
        """Union of one or more bounding boxes with this box"""        
        bblist = tolist(bb)        
        assert all([isinstance(bb, BoundingBox) for bb in bblist]), "Invalid BoundingBox() input"
        self._xmin = min([bb.xmin() for bb in bblist] + [self.xmin()])
        self._ymin = min([bb.ymin() for bb in bblist] + [self.ymin()])
        self._xmax = max([bb.xmax() for bb in bblist] + [self.xmax()])
        self._ymax = max([bb.ymax() for bb in bblist] + [self.ymax()])
        return self

    def isinside(self, bb):
        """Is this boundingbox fully within the provided bounding box?"""
        assert isinstance(bb, BoundingBox)
        return self.hasintersection(bb) and self.cover(bb) == 1.0
        
    def ispointinside(self, p):
        """Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?"""
        assert len(p) == 2, "Invalid 2D point=(x,y) input"""
        return (p[0] >= self._xmin) and (p[1] >= self._ymin) and (p[0] <= self._xmax) and (p[1] <= self._ymax)

    def dilate(self, scale=1):
        """Change scale of bounding box keeping centroid constant"""
        assert isnumber(scale), "Invalid input"
        w = (self._xmax - self._xmin)
        h = (self._ymax - self._ymin)
        c = self.centroid()
        old_x = self._xmin
        old_y = self._ymin
        new_x = (float(w) / 2.0) * scale
        new_y = (float(h) / 2.0) * scale
        self._xmin = c[0] - new_x
        self._ymin = c[1] - new_y
        self._xmax = c[0] + new_x
        self._ymax = c[1] + new_y
        return self

    def dilate_height(self, scale=1):
        """Change scale of bounding box in y direction keeping centroid constant"""
        h = self.height()
        c = self.centroid()
        self._ymin = c[1] - (float(h) / 2.0) * scale
        self._ymax = c[1] + (float(h) / 2.0) * scale
        return self

    def dilate_width(self, scale=1):
        """Change scale of bounding box in x direction keeping centroid constant"""
        w = self._xmax - self._xmin
        c = self.centroid()
        self._xmin = c[0] - (float(w) / 2.0) * scale
        self._xmax = c[0] + (float(w) / 2.0) * scale
        return self

    def top(self, dy):
        """Make top of box taller (closer to top of image) by an offset dy"""
        self._ymin = self._ymin - dy
        return self

    def bottom(self, dy):
        """Make bottom of box taller (closer to bottom of image) by an offset dy"""
        self._ymax = self._ymax + dy
        return self

    def left(self, dx):
        """Make left of box wider (closer to left side of image) by an offset dx"""
        self._xmin = self._xmin - dx
        return self

    def right(self, dx):
        """Make right of box wider (closer to right side of image) by an offset dx"""
        self._xmax = self._xmax + dx
        return self

    def rescale(self, scale=1):
        """Multiply the box corners by a scale factor"""
        self._xmin = scale * self._xmin
        self._ymin = scale * self._ymin
        self._xmax = scale * self._xmax
        self._ymax = scale * self._ymax
        return self

    def scalex(self, scale=1):
        """Multiply the box corners in the x dimension by a scale factor"""
        self._xmin = scale * self._xmin
        self._xmax = scale * self._xmax
        return self

    def scaley(self, scale=1):
        """Multiply the box corners in the y dimension by a scale factor"""
        self._ymin = scale * self._ymin
        self._ymax = scale * self._ymax
        return self

    def resize(self, width, height):
        """Change the aspect ratio width and height of the box"""
        self.setwidth(width)
        self.setheight(height)
        return self

    def rot90cw(self, H, W):
        """Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align"""
        (x,y,w,h) = self.xywh()
        (blx, bly) = self.bottomleft()
        return self.xywh((H - bly, blx, h, w))

    def rot90ccw(self, H, W):
        """Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align"""
        (x,y,w,h) = self.xywh()
        (urx, ury) = self.upperright()
        return self.xywh((ury, W - urx, h, w))

    def fliplr(self, img=None, width=None):
        """Flip the box left/right consistent with fliplr of the provided img (or consistent with the image width)"""
        if img is not None:
            assert isnumpy(img), "Invalid numpy image input"
            width = img.shape[1]
        else:
            assert isnumber(width), "Invalid width"
        (x,y,w,h) = self.xywh()
        self._xmin = width - self._xmax
        self._xmax = self._xmin + w
        return self

    def flipud(self, img=None, height=None):
        """Flip the box up/down consistent with flipud of the provided img (or consistent with the image height)"""
        if img is not None:
            assert isnumpy(img), "Invalid numpy image input"
            height = img.shape[0]
        else:
            assert height is not None and isnumber(height), "Invalid height"
        (x,y,w,h) = self.xywh()
        self._ymin = height - self._ymax
        self._ymax = self._ymin + h
        return self

    def imscale(self, im):
        """Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image"""
        w = (1.0 / float(im.width()))
        h = (1.0 / float(im.height()))
        self._xmin = w * self._xmin
        self._ymin = h * self._ymin
        self._xmax = w * self._xmax
        self._ymax = h * self._ymax
        return self

    def maxsquare(self):
        """Set the bounding box to be square by setting width and height to the maximum dimension of the box, keeping centroid constant"""
        (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)                            
        if width != height:
            dim = float(max(width, height))
            c = self.centroid()
            self._xmin = c[0] - (dim / 2.0)
            self._ymin = c[1] - (dim / 2.0)
            self._xmax = c[0] + (dim / 2.0)
            self._ymax = c[1] + (dim / 2.0)
        return self

    def maxsquareif(self, do):
        return self.maxsquare() if do else self

    def issquare(self):
        return np.allclose(self.bbheight(), self.bbwidth())

    def iseven(self):
        """Are all corners even number integers?"""
        return (isinstance(self.xmin(), int) and self.xmin() % 2 == 0 and
                isinstance(self.ymin(), int) and self.ymin() % 2 == 0 and
                isinstance(self.xmax(), int) and self.xmax() % 2 == 0 and
                isinstance(self.ymax(), int) and self.ymax() % 2 == 0)

    def even(self):
        """Force all corners to be even number integers.  This is helpful for FFMPEG crop filters."""
        self.int()
        self._xmin = (self._xmin // 2) * 2
        self._ymin = (self._ymin // 2) * 2
        self._xmax = (self._xmax // 2) * 2
        self._ymax = (self._ymax // 2) * 2
        return self

    def minsquare(self):
        """Set the bounding box to be square by setting width and height to the minimum dimension of the box, keeping centroid constant"""
        (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)                            
        if width != height:
            dim = float(min(width, height))
            c = self.centroid()
            self._xmin = c[0] - (dim / 2.0)
            self._ymin = c[1] - (dim / 2.0)
            self._xmax = c[0] + (dim / 2.0)
            self._ymax = c[1] + (dim / 2.0)
        return self

    def hasoverlap(self, img=None, width=None, height=None):
        """Does the bounding box intersect with the provided image rectangle?"""
        if img is not None:
            assert isnumpy(img), "Invalid image input"
            (width, height) = (img.shape[1], img.shape[0])
        else:
            assert width is not None and height is not None, "Invalid width and height - both must be provided"
            assert isnumber(width) and isnumber(height), "Invalid width and height - both must be numbers"
        return self.area_of_intersection(BoundingBox(xmin=0, ymin=0, width=width, height=height)) > 0

    def isinterior(self, width=None, height=None):
        """Is this boundingbox fully within the provided image rectangle?"""
        return self.isinside(imagebox((height, width)))
        
    def iminterior(self, W, H):
        """Transform bounding box to be interior to the image rectangle with shape (W,H).  
           Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit
        """        
        assert self.intersection(BoundingBox(xmin=0, ymin=0, width=W, height=H)).area() > 0, "Bounding box must intersect image rectangle"
        self.translate(dx=0 if self.xmin()>0 else -self.xmin(),
                       dy=0 if self.ymin()>0 else -self.ymin())
        self.translate(dx=0 if self.xmax()<W else -(W-self.xmax()),
                       dy=0 if self.ymax()<H else -(H-self.ymax()))
        return self.imclip(width=W, height=H)
        
    def imclip(self, img=None, width=None, height=None):
        """Clip bounding box to image rectangle [0,0,width,height] or img.shape=(width, height) and, throw an exception on an invalid box"""
        if img is not None:
            assert isnumpy(img), "Invalid numpy image input"
            (height, width) = (img.shape[0], img.shape[1])
        else:
            assert width is not None and height is not None, "Invalid width and height - both must be provided"
            assert isnumber(width) and isnumber(height), "Invalid width and height - both must be numbers"
        return self.intersection(BoundingBox(xmin=0, ymin=0, width=width, height=height), strict=True)

    def imclipshape(self, W, H):
        """Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box"""
        return self.imclip(width=W, height=H)

    def convexhull(self, fr):
        """Given a set of points [[x1,y1],[x2,xy],...], return the bounding rectangle, typecast to float"""
        self._xmin = float(np.min(fr[:,0]))
        self._ymin = float(np.min(fr[:,1]))
        self._xmax = float(np.max(fr[:,0]))
        self._ymax = float(np.max(fr[:,1]))
        return self

    def aspectratio(self):
        """Return the aspect ratio (width/height) of the box"""
        (height, width) = (self._ymax-self._ymin, self._xmax-self._xmin)                            
        assert height > 0
        return float(width) / float(height)

    def shape(self):
        """Return the (height, width) tuple for the box shape"""
        return (self._ymax-self._ymin, self._xmax-self._xmin)                            

    def mindimension(self):
        """Return min(width, height) typecast to float"""
        return float(np.min(self.shape()))

    def mindim(self):
        """Return min(width, height) typecast to float"""
        return float(np.min(self.shape()))

    def maxdim(self):
        """Return max(width, height) typecast to float"""
        return float(np.max(self.shape())) 
    
    def ellipse(self):
        """Convert the boundingbox to a vipy.geometry.Ellipse object"""
        (xcenter,ycenter) = self.centroid()
        return Ellipse(self.bbwidth() / 2.0, self.bbheight() / 2.0, xcenter, ycenter, 0)

    def average(self, other):
        """Compute the average bounding box between self and other.  Other may be a singleton bounding box or a list of bounding boxes"""
        assert all([isinstance(bb, BoundingBox) for bb in tolist(other)]), "Invalid input - must be BoundingBox"        
        return self.ulbr(np.mean( [self.ulbr()] + [bb.ulbr() for bb in tolist(other)], axis=0))

    def averageshape(self, other):
        """Compute the average bounding box width and height between self and other.  Other may be a singleton bounding box or a list of bounding boxes"""
        assert all([isinstance(bb, BoundingBox) for bb in tolist(other)]), "Invalid input - must be BoundingBox"        
        (xmin, ymin, xmax, ymax) = np.mean( [self.ulbr()] + [bb.ulbr() for bb in tolist(other)], axis=0)
        self.setwidth(xmax-xmin)
        self.setheight(ymax-ymin)        
        return self

    def medianshape(self, other):
        """Compute the median bounding box width and height between self and other.  Other may be a singleton bounding box or a list of bounding boxes"""
        assert all([isinstance(bb, BoundingBox) for bb in tolist(other)]), "Invalid input - must be BoundingBox"        
        (height, width) = np.median( [self.shape()] + [bb.shape() for bb in tolist(other)], axis=0)
        self.setwidth(width)
        self.setheight(height)
        return self

    def shapedist(self, other):
        """L1 distance between (width,height) of two boxes"""
        assert isinstance(other, BoundingBox), "Invalid input - must be BoundingBox()"                
        return np.abs(self.bbwidth()-other.bbwidth())  + np.abs(self.bbheight()-other.bbheight())

    def affine(self, A):
        """Apply an 2x3 affine transformation to the box centroid.  This operation preserves an axis aligned bounding box for an arbitrary affine transform."""
        assert isnumpy(A) and A.shape == (2,3), "A must be a 2x3 affine transformation matrix"
        return self.centroid(np.dot(A, homogenize(np.array(self.centroid()))))

    def projective(self, A):
        """Apply an 3x3 affine transformation to the box centroid.  This operation preserves an axis aligned bounding box for an arbitrary affine transform."""
        assert isnumpy(A) and A.shape == (3,3), "A must be a 3x3 affine transformation matrix"
        return self.centroid(dehomogenize(np.dot(A, homogenize(np.array(self.centroid())))))
    
    def crop(self, img):
        """Crop an HxW 2D numpy image, HxWxC 3D numpy image, or NxHxWxC 4D numpy image array using this bounding box applied to HxW dimensions.  Sets bounding box to integer coordinates"""
        assert isnumpy(img) and img.ndim in [2,3,4]
        if img.ndim == 2:
            return img[self.int().ymin():self.ymax(), self.xmin():self.xmax()]  # HxW
        elif img.ndim == 3:
            return img[self.ymin():self.ymax(), self.xmin():self.xmax(), :]  # HxWxC
        else: 
            return img[:, self.ymin():self.ymax(), self.xmin():self.xmax(), :]  # NxHxWxC


class Ellipse():
    __slots__ = ['_major', '_minor', '_xcenter', '_ycenter', '_phi']
    def __init__(self, semi_major, semi_minor, xcenter, ycenter, phi):
        """Ellipse parameterization, for length of semimajor (half width of ellipse) and semiminor axis (half height), center point and angle phi in radians"""
        self._major = semi_major
        self._minor = semi_minor
        self._xcenter = xcenter
        self._ycenter = ycenter
        self._phi = phi

    def __repr__(self):
        return str('<vipy.geometry.ellipse: semimajor=%s, semiminor=%s, xcenter=%s, ycenter=%s, phi=%s (rad)>' % (self._major, self._minor, self._xcenter, self._ycenter, self._phi))

    def dict(self):
        return {'semimajor':self._major, 'semiminor':self._minor, 'xcenter':self._xcenter, 'ycenter':self._ycenter, 'phi':self._phi}
    
    def area(self):
        """Area of ellipse"""
        return math.pi * self._major * self._minor

    def center(self):
        """Return centroid"""
        return (self._xcenter, self._ycenter)

    def centroid(self):
        """Alias for center"""
        return self.center()

    def axes(self):
        """Return the (major,minor) axis lengths"""
        return (self._major, self._minor)

    def angle(self):
        """Return the angle phi (in degrees)"""
        return (self._phi * 180 / math.pi)

    def rescale(self, scale):
        """Scale ellipse by scale factor"""
        assert isnumber(scale), "Invalid input"
        self._major *= scale
        self._minor *= scale
        self._xcenter *= scale
        self._ycenter *= scale
        return self

    def boundingbox(self):
        """ Estimate an equivalent bounding box based on scaling to a common area.
        Note, this does not factor in rotation.
        (c*l)*(c*w) = a_e  --> c = sqrt(a_e / a_r) """
        assert self._phi == 0, "This function does not currently factor in rotation"

        bbox = BoundingBox(width=2 * self._major, height=2 * self._minor, xcentroid=self._xcenter, ycentroid=self._ycenter)
        a_r = bbox.area()
        c = (self.area() / a_r) ** 0.5
        bbox2 = bbox.clone().dilate(c)
        return bbox2

    def inside(self, x, y=None):
        """Return true if a point p=(x,y) is inside the ellipse"""
        p = (x,y) if y is not None else x
        assert len(p) == 2, "Invalid input"
        assert self._phi == 0, "inside only currently supported for phi=0"
        return ((np.square(p[0] - self._xcenter) / np.square(self._major)) + (np.square(p[1] - self._ycenter) / np.square(self._minor))) <= 1

    def mask(self):
        """Return a binary mask of size equal to the bounding box such that the pixels correspond to the interior of the ellipse"""
        (H,W) = (int(np.round(2 * self._minor)), int(np.round(2 * self._major)))
        img = np.zeros((H,W), dtype=bool)
        for (y,x) in product(range(0,H), range(0,W)):
            img[y,x] = self.inside(x,y)
        return img
