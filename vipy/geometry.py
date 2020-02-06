import numpy as np
import math
import numpy.linalg
import scipy.spatial
from itertools import product
from vipy.util import try_import, istuple, isnumpy
from vipy.math import isnumber

def covariance_to_ellipse(cov):
    """2x2 covariance matrix to rotated bounding box"""
    (d,V) = numpy.linalg.eig(cov)
    return (d[0], d[1], math.atan2(V[1,0],V[0,0]))  # (major_axis_len, minor_axis_len, angle_in_radians)

def dehomogenize(p):
    return np.float32(p[0:-1, :]) / np.float32(p[-1,:])

def homogenize(p):
    return np.float32(np.vstack( (p, np.ones((1,p.shape[1]))) ))

def apply_homography(H,p):
    return dehomogenize(H*homogenize(p))

def similarity_imtransform2D(c=(0,0), r=0, s=1):
    try_import(cv2, 'opencv-python')
    import cv2
    deg = r * 180. / math.pi
    A = cv2.getRotationMatrix2D(c, deg, s)
    return A

def similarity_imtransform(txy=(0,0), r=0, s=1):
    R = np.mat([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0,0,1]])
    S = np.mat([[s,0,0], [0, s, 0], [0,0,1]])
    T = np.mat([[0,0,txy[0]], [0,0,txy[1]], [0,0,0]])
    return S*R + T  # composition


def affine_imtransform(txy=(0,0), r=0, sx=1, sy=1, kx=0, ky=0):
    R = np.mat([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0,0,1]])
    S = np.mat([[sx,0,0], [0, sy, 0], [0,0,1]])
    K = np.mat([[1,ky,0], [kx,1,0], [0,0,1]])
    T = np.mat([[0,0,txy[0]], [0,0,txy[1]], [0,0,0]])
    return K*S*R + T  # composition

def random_affine_imtransform(txy=((0,0),(0,0)), r=(0,0), sx=(1,1), sy=(1,1), kx=(0,0), ky=(0,0)):
    return affine_imtransform(txy=(uniform_random_in_range(txy[0]), uniform_random_in_range(txy[1])),
                              r=uniform_random_in_range(r),
                              sx=uniform_random_in_range(sx),
                              sy=uniform_random_in_range(sy),
                              kx=uniform_random_in_range(kx),
                              ky=uniform_random_in_range(ky))

def imtransform2D(im, A):
    try_import(cv2, 'opencv-python'); import cv2    
    return cv2.warpAffine(im, A, (im.shape[1], im.shape[0]))

def imtransform(im, A):
    try_import(cv2, 'opencv-python'); import cv2        
    return cv2.warpPerspective(im, A, (im.shape[1], im.shape[0]))

def frame_to_bbox(fr):
    """ bbox = [ (xmin,ymin,width,height), ... ] """
    return [ (x[0]-x[2]/2, x[1]-x[2]/2, x[0]+x[2]/2, x[1]+x[2]/2) for x in fr]

def sub2ind(i, j, rows):
    return (i*rows + j)

def sqdist(d_obs, d_ref):
    """Given d_obs of size (MxD) ad d_ref of size (NxD) return (MxN) matrix of pairwise euclidean distances"""
    return scipy.spatial.distance.cdist(np.array(d_obs), np.array(d_ref), 'euclidean')  # small number of descriptors only

def normalize(x):
    return x / (np.linalg.norm(x.astype(np.float64))+1E-16)


class BoundingBox():
    """Core bounding box class with flexible constructors in this priority order:
          (xmin,ymin,xmax,ymax)
          (xmin,ymin,width,height)
          (centroid[0],centroid[1],width,height)
          (xcentroid,ycentroid,width,height)
          bounding rectangle of binary mask image"""
    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, centroid=None, xcentroid=None, ycentroid=None, width=None, height=None, mask=None):

        if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            if not (isnumber(xmin) and isnumber(ymin) and isnumber(xmax) and isnumber(ymax)):
                raise ValueError('Box coordinates must be integers or floats')
            self._xmin = float(xmin)
            self._ymin = float(ymin)
            self._xmax = float(xmax)
            self._ymax = float(ymax)
        elif xmin is not None and ymin is not None and width is not None and height is not None:
            if not (isnumber(xmin) and isnumber(ymin) and isnumber(width) and isnumber(height)):
                raise ValueError('Box coordinates must be integers or floats')            
            self._xmin = float(xmin)
            self._ymin = float(ymin)
            self._xmax = self._xmin + float(width)
            self._ymax = self._ymin + float(height)
        elif centroid is not None and width is not None and height is not None:
            if not (istuple(centroid) and len(centroid) == 2 and isnumber(centroid[0]) and isnumber(centroid[1]) and isnumber(width) and isnumber(height)):
                raise ValueError('Invalid box coordinates')
            self._xmin = float(centroid[0]) - float(width)/2.0
            self._ymin = float(centroid[1]) - float(height)/2.0
            self._xmax = float(centroid[0]) + float(width)/2.0
            self._ymax = float(centroid[1]) + float(height)/2.0                                                
        elif xcentroid is not None and ycentroid is not None and width is not None and height is not None:
            if not (isnumber(xcentroid) and isnumber(ycentroid) and isnumber(width) and isnumber(height)):
                raise ValueError('Box coordinates must be integers or floats')                        
            self._xmin = float(xcentroid) - (float(width)/2.0)
            self._ymin = float(ycentroid) - (float(height)/2.0)
            self._xmax = float(xcentroid) + (float(width)/2.0)
            self._ymax = float(ycentroid) + (float(height)/2.0)                                                
        elif mask is not None:
            # Bounding rectangle of non-zero pixels in a binary mask image
            if not isnumpy(mask) or np.sum(mask)==0:
                raise ValueError('Mask input must be numpy array with at least one non-zero entry')            
            imx = np.sum(mask, axis=0)
            imy = np.sum(mask, axis=1)            
            self._xmin = np.argwhere(imx > 0)[0]
            self._ymin = np.argwhere(imy > 0)[0]
            self._xmax = np.argwhere(imx > 0)[-1]
            self._ymax = np.argwhere(imy > 0)[-1]
        else:
            raise ValueError('invalid constructor input')

    def clone(self):
        return BoundingBox(xmin=self._xmin, xmax=self._xmax, ymin=self._ymin, ymax=self._ymax)
        
    def __eq__(self, other):
        """Bounding box equality"""
        return self.xmin()==other.xmin() and self.xmax()==other.xmax() and self.ymin()==other.ymin() and self.ymax()==other.ymax()

    def __neq__(self, other):
        """Bounding box non-equality"""
        return not self.__eq__(other)

    def __repr__(self):
        return str('<vipy.geometry.boundingbox: xmin=%s, ymin=%s, width=%s, height=%s>'% (self.xmin(), self.ymin(), self.width(), self.height()))

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

    def lowerleft(self):
        """Return the (x,y) lower left corner coordinate of the box"""
        return (self.xmin(), self.ymax())
    
    def upperright(self):
        """Return the (x,y) upper right corner coordinate of the box"""
        return (self.xmax(), self.ymin())

    def lowerright(self):
        """Return the (x,y) lower right corner coordinate of the box"""
        return (self.xmax(), self.ymax())
    
    def invalid(self):
        """Is the box a valid bounding box?"""
        is_undefined = np.isnan(self._xmin) or np.isnan(self._ymin) or np.isnan(self._xmax) or np.isnan(self._ymax)
        is_degenerate =  ((self._xmax-self._xmin)<1) or ((self._ymax-self._ymin)<1) or (self._xmin >= self._xmax) or (self._ymin >= self._ymax)
        return is_undefined or is_degenerate

    def int(self):
        """Convert corners to integer with rounding"""
        self._xmin = int(np.round(self._xmin))
        self._ymin = int(np.round(self._ymin))
        self._xmax = int(np.round(self._xmax))
        self._ymax = int(np.round(self._ymax))
        return self
    
    def translate(self, dx=0, dy=0):
        self._xmin = self._xmin + dx
        self._ymin = self._ymin + dy
        self._xmax = self._xmax + dx
        self._ymax = self._ymax + dy
        return self
    def offset(self, dx=0, dy=0):
        """Alias for translate"""
        return self.translate(dx, dy)
    
    def valid(self):
        return not self.invalid()

    def isvalid(self):
        return self.valid()

    def isdegenerate(self):
        return self.invalid()
    
    def width(self):
        return self._xmax - self._xmin

    def setwidth(self, w):
        """Set new width keeping centroid constant"""        
        if w <= 0:
            raise ValueError('invalid width')
        worig = self.width()        
        self._xmax += (w - worig) / 2.0
        self._xmin -= (w - worig) / 2.0        
        return self
    
    def setheight(self, h):
        """Set new height keeping centroid constant"""
        if h <= 0:
            raise ValueError('invalid height')
        horig = self.height()
        self._ymax += (h - horig) / 2.0
        self._ymin -= (h - horig) / 2.0        
        return self
        
    def height(self):
        return self._ymax - self._ymin
    
    def centroid(self):
        """(x,y) tuple of centroid"""
        return [self._xmin + (float(self.width())/2.0), self._ymin + (float(self.height())/2.0)]
            
    def x_centroid(self):
        return self.centroid()[0]

    def y_centroid(self):
        return self.centroid()[1]
        
    def area(self):
        return self.width() * self.height()

    def to_xywh(self, xywh=None):
        """Return bounding box corners as [x,y,width,height] format"""
        if xywh is None:
            return [self._xmin, self._ymin, self.width(), self.height()]
        else:
            self._xmin = xywh[0]
            self._ymin = xywh[1]
            self._xmax = self._xmin + xywh[2]
            self._ymax = self._ymin + xywh[3]
            return self

    def xywh(self, xywh_=None):
        """Alias for to_xywh"""
        return self.to_xywh(xywh_)
    
    def dx(self, bb):
        """Offset bounding box by same xmin as provided box"""
        return bb._xmin - self._xmin

    def dy(self, bb):
        """Offset bounding box by ymin of provided box"""        
        return bb._ymin - self._ymin

    def sqdist(self, bb):
        """Squared Euclidean distance between upper left corners of two bounding boxes"""
        return np.power(self.dx(bb), 2.0) + np.power(self.dy(bb), 2.0)

    def dist(self, bb):
        """Distance between centroids of two bounding boxes"""
        return np.sqrt(np.sum(np.square(np.array(bb.centroid()) - np.array(self.centroid()))))
    
    def iou(self, bb):
        """area of intersection / area of union"""
        if bb is None or bb.invalid() or self.invalid():
            return 0.0
        w = min(self.xmax(), bb.xmax()) - max(self.xmin(), bb.xmin());
        h = min(self.ymax(), bb.ymax()) - max(self.ymin(), bb.ymin());
        if ((w < 0) or (h < 0)):
            iou = 0;   # invalid (no overlap)
        else:
            area_intersection = w*h;
            area_union = (self.area() + bb.area() - area_intersection);
            iou = area_intersection / area_union;
        return iou;

    def intersection_over_union(self, bb):
        """Alias for iou"""
        return self.iou(bb)
    
    def area_of_intersection(self, bb):
        """area of intersection"""
        if bb.invalid() or self.invalid():
            return 0.0
        w = min(self.xmax(), bb.xmax()) - max(self.xmin(), bb.xmin());
        h = min(self.ymax(), bb.ymax()) - max(self.ymin(), bb.ymin());
        if ((w < 0) or (h < 0)):
            aoi = 0;   # invalid (no overlap)
        else:
            aoi = w*h;
        return aoi

    def cover(self, bb):
        """Fraction of this bounding box covered by other bbox"""
        return self.area_of_intersection(bb) / float(self.area())

    def intersection(self, bb, strict=True):
        """Intersection of two bounding boxes, throw an error on degeneracy if strict"""
        self._xmin = max(bb.xmin(), self.xmin())
        self._ymin = max(bb.ymin(), self.ymin())
        self._xmax = min(bb.xmax(), self.xmax())
        self._ymax = min(bb.ymax(), self.ymax())
        if strict and self.isdegenerate():
            raise ValueError('Degenerate intersection')
        return self
                    
    def union(self, bb):
        """Union of two bounding boxes"""
        self._xmin = min(bb.xmin(), self.xmin())
        self._ymin = min(bb.ymin(), self.ymin())
        self._xmax = max(bb.xmax(), self.xmax())
        self._ymax = max(bb.ymax(), self.ymax())
        return self
        
    def inside(self, p):
        """Is the 2D point p=(x,y) inside the bounding box?"""
        return (p[0] >= self._xmin) and (p[1] >= self._ymin) and (p[0] <= self._xmax) and (p[1] <= self._ymax)

    def dilate(self, scale=1):
        """Change scale of bounding box keeping centroid constant"""
        w = self.width()
        h = self.height()
        c = self.centroid()
        old_x = self._xmin
        old_y = self._ymin
        new_x = (float(w)/2.0) * scale
        new_y = (float(h)/2.0) * scale
        self._xmin = c[0] - new_x
        self._ymin = c[1] - new_y
        self._xmax = c[0] + new_x
        self._ymax = c[1] + new_y
        return self

    def dilate_height(self, scale=1):
        """Change scale of bounding box in y direction keeping centroid constant"""
        h = self.height()
        c = self.centroid()
        self._ymin = c[1]-(float(h)/2.0)*scale
        self._ymax = c[1]+(float(h)/2.0)*scale
        return self

    def dilate_width(self, scale=1):
        """Change scale of bounding box in x direction keeping centroid constant"""
        w = self.width()
        c = self.centroid()
        self._xmin = c[0]-(float(w)/2.0)*scale
        self._xmax = c[0]+(float(w)/2.0)*scale
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
        self.setheigh(height)
        return self

    def rot90cw(self, H, W):
        """Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align"""
        (x,y,w,h) = self.xywh()
        (blx, bly) = self.lowerleft()        
        return self.xywh( (H-bly, blx, h, w) )

    def rot90ccw(self, H, W):
        """Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align"""
        (x,y,w,h) = self.xywh()
        (urx, ury) = self.upperright()        
        return self.xywh( (ury, W-urx, h, w) )
    
    def fliplr(self, img):
        """Flip the box left/right consistent with fliplr of the provided img"""
        assert isnumpy(img), "Invalid image input"
        (x,y,w,h) = self.to_xywh()
        self._xmin = img.shape[1] - xmax
        self._xmax = self._xmin + w

        
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
        """Set the bounding box to be square by dilating the minimum dimension, keeping centroid constant"""
        w = max(self.width(), self.height())
        h = max(self.height(), self.width())
        c = self.centroid()
        self._xmin = c[0]-(float(w)/2.0)
        self._ymin = c[1]-(float(h)/2.0)
        self._xmax = c[0]+(float(w)/2.0)
        self._ymax = c[1]+(float(h)/2.0)
        return self

    def hasoverlap(self, img):
        """Does the bounding box intersect with the provided image rectangle?"""
        assert isnumpy(img), "Invalid image input"        
        return self.area_of_intersection(BoundingBox(xmin=0, ymin=0, xmax=img.shape[1]-1, ymax=img.shape[0]-1)) > 0
        
    def imclip(self, img):
        """Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box"""
        assert isnumpy(img), "Invalid image input"        
        self.intersection(BoundingBox(xmin=0, ymin=0, xmax=img.shape[1]-1, ymax=img.shape[0]-1))
        return self

    def imclipshape(self, W, H):
        """Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box"""
        self.intersection(BoundingBox(xmin=0, ymin=0, xmax=W-1, ymax=H-1))
        return self
    
    def convexhull(self, fr):
        """Given a set of points [[x1,y1],[x2,xy],...], return the bounding rectangle"""
        self._xmin = np.min(fr[:,0])
        self._ymin = np.min(fr[:,1])
        self._xmax = np.max(fr[:,0])
        self._ymax = np.max(fr[:,1])
        return self

    def aspectratio(self):
        """Return the aspect ratio (width/height) of the box"""
        return float(self.width()) / float(self.height())

    def shape(self):
        """Return the (height, width) tuple for the box shape"""
        return (self.height(), self.width())

    def mindimension(self):
        """Return min(width, height)"""
        return np.min(self.shape())

    def to_ellipse(self):
        (xcenter,ycenter) = self.centroid()
        return Ellipse(self.width()/2.0, self.height()/2.0, xcenter, ycenter,0)
    
    
class Ellipse():
    def __init__(self, semi_major, semi_minor, xcenter, ycenter, phi):
        """Ellipse parameterization, for length of semimajor (half width of ellipse) and semiminor axis (half height), center point and angle phi in radians"""
        self.major = semi_major
        self.minor = semi_minor
        self.center_x = xcenter
        self.center_y = ycenter
        self.phi = phi

    def __repr__(self):
        return str('<vipy.geometry.ellipse: semimajor=%s, semiminor=%s, xcenter=%s, ycenter=%s, phi=%s (rad)>'% (self.major, self.minor, self.center_x, self.center_y, self.phi))


    def area(self):
        return math.pi * self.major * self.minor

    def center(self):
        return (int(self.center_x), int(self.center_y))

    def axes(self):
        return (int(self.major), int(self.minor))

    def angle(self):
        """in degrees"""
        return int(self.phi * 180 / math.pi)

    def rescale(self, scale):
        self.major *= scale
        self.minor *= scale
        self.center_x *= scale
        self.center_y *= scale
        return self

    def boundingbox(self):
        """ Estimate an equivalent bounding box based on scaling to a common area.
        Note, this does not factor in rotation.
        (c*l)*(c*w) = a_e  --> c = \sqrt(a_e / a_r) """

        bbox = BoundingBox(width=2*self.major, height=2*self.minor, xcentroid=self.center_x, ycentroid=self.center_y)
        a_r = bbox.area()
        c = (self.area() / a_r) ** 0.5
        bbox2 = bbox.clone().dilate(c)
        return bbox2

    def inside(self, x, y=None):
        """Return true if a point p=(x,y) is inside the ellipse"""
        p = (x,y) if y is not None else x
        if (self.phi != 0):
            raise ValueError('FIXME: inside only supported for phi=0')
        return ((np.square(p[0] - self.center_x) / np.square(self.major)) + (np.square(p[1] - self.center_y) / np.square(self.minor))) <= 1

    def mask(self):
        """Return a binary mask of size equal to the bounding box such that the pixels correspond to the interior of the ellipse"""
        (H,W) = (int(np.round(2*self.minor)), int(np.round(2*self.major)))
        img = np.zeros( (H,W) )
        for (y,x) in product(range(0,H), range(0,W)):
            img[y,x] = self.inside(x,y)
        return img

