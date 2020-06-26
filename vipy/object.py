import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist, chunklistwithoverlap
import uuid
import copy
import warnings
import scipy.interpolate


class Detection(BoundingBox):
    """vipy.object.Detection class
    
    This class represent a single object detection in the form a bounding box with a label and confidence.
    The constructor of this class follows a subset of the constructor patterns of vipy.geometry.BoundingBox

    >>> d = vipy.object.Detection(category='Person', xmin=0, ymin=0, width=50, height=100)
    >>> d = vipy.object.Detection(label='Person', xmin=0, ymin=0, width=50, height=100)  # "label" is an alias for "category"
    >>> d = vipy.object.Detection(label='John Doe', shortlabel='Person', xmin=0, ymin=0, width=50, height=100)  # shortlabel is displayed
    >>> d = vipy.object.Detection(label='Person', xywh=[0,0,50,100])

    """

    def __init__(self, label=None, xmin=None, ymin=None, width=None, height=None, xmax=None, ymax=None, confidence=None, xcentroid=None, ycentroid=None, category=None, xywh=None, shortlabel=None, attributes=None):
        super(Detection, self).__init__(xmin=xmin, ymin=ymin, width=width, height=height, xmax=xmax, ymax=ymax, xcentroid=xcentroid, ycentroid=ycentroid, xywh=xywh)
        assert not (label is not None and category is not None), "Constructor requires either label or category kwargs, not both"
        self._id = uuid.uuid1().hex        
        self._label = category if category is not None else label
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._confidence = float(confidence) if confidence is not None else confidence
        self.attributes = attributes if attributes is not None else {}

    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.isvalid():
            strlist.append('bbox=(xmin=%1.1f, ymin=%1.1f, width=%1.1f, height=%1.1f)' %
                           (self.xmin(), self.ymin(),self.width(), self.height()))
        if self._confidence is not None:
            strlist.append('conf=%1.3f' % self.confidence())
        return str('<vipy.object.detection: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        """Detection equality when bounding boxes and categories are equivalent"""
        return isinstance(other, Detection) and self.xywh() == other.xywh() and self.category() == other.category()

    def __str__(self):
        return self.__repr__()

    def dict(self):
        return {'id':self._id, 'label':self.category(), 'shortlabel':self.shortlabel() ,'boundingbox':super(Detection, self).dict(),
                'attributes':self.attributes,  # these may be arbitrary user defined objects
                'confidence':self._confidence}
                    
    def nocategory(self):
        self._label = None
        return self

    def category(self, category=None):
        """Update the category of the detection"""
        if category is None:
            return self._label
        else:
            self._label = str(category)  # coerce to string
            return self

    def shortlabel(self, label=None):
        """A optional shorter label string to show in the visualizations, defaults to category()"""        
        if label is not None:
            self._shortlabel = str(label)  # coerce to string
            return self
        else:
            return self._shortlabel if self._shortlabel is not None else self.category()

    def label(self, label):
        """Alias for category"""
        return self.category(label)

    def id(self):
        return self._id

    def clone(self):
        return copy.deepcopy(self)

    def confidence(self):
        return self._confidence
    
    
class Track(object):
    """vipy.object.Track class
    
    A track represents one or more labeled bounding boxes of an object instance through time.  A track is defined as a finite set of labeled boxes observed 
    at keyframes, which are discrete observations of this instance.  Each keyframe has an associated vipy.geometry.BoundingBox() which defines the spatial bounding box
    of the instance in this keyframe.  The kwarg "interpolation" defines how the track is interpolated between keyframes, and the kwarg "boundary" defines how the 
    track is interpolated outside the (min,max) of the keyframes.  

    Valid constructors are:

    >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person')
    >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', interpolation='linear')
    >>> t = vipy.object.Track(keyframes=[10,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', boundary='strict')

    Tracks can be constructed incrementally:

    >>> t = vipy.object.Track('Person')
    >>> t.add(0, vipy.geometry.BoundingBox(0,0,10,10))
    >>> t.add(100, vipy.geometry.BoundingBox(0,0,20,20))

    Tracks can be resampled at a new framerate, as long as the framerate is known when the keyframes are extracted

    >>> t.framerate(newfps)

    """

    def __init__(self, keyframes, boxes, category=None, label=None, confidence=None, framerate=None, interpolation='linear', boundary='strict', shortlabel=None, attributes=None, trackid=None):

        keyframes = tolist(keyframes)
        boxes = tolist(boxes)        
        assert isinstance(keyframes, tuple) or isinstance(keyframes, list), "Keyframes are required and must be tuple or list"
        assert isinstance(boxes, tuple) or isinstance(boxes, list), "Keyframe boundingboxes are required and must be tuple or list"
        assert all([isinstance(bb, BoundingBox) for bb in boxes]), "Keyframe bounding boxes must be vipy.geometry.BoundingBox objects"
        assert all([bb.isvalid() for bb in boxes]), "All keyframe bounding boxes must be valid"        
        assert not (label is not None and category is not None), "Constructor requires either label or category kwargs, not both"                
        assert len(keyframes) == len(boxes), "Boxes and keyframes must be the same length, there must be a one to one mapping of frames to boxes"
        assert boundary in set(['extend', 'strict']), "Invalid interpolation boundary - Must be ['extend', 'strict']"
        assert interpolation in set(['linear']), "Invalid interpolation - Must be ['linear']"
                
        self._id = uuid.uuid1().hex if trackid is None else trackid
        self._label = category if category is not None else label
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._framerate = framerate
        self._interpolation = interpolation
        self._boundary = boundary
        self.attributes = attributes if attributes is not None else {}        
        self._keyframes = keyframes
        self._keyboxes = boxes
        
        # Sorted increasing frame order
        if len(keyframes) > 0 and len(boxes) > 0:
            (keyframes, boxes) = zip(*sorted([(f,bb) for (f,bb) in zip(keyframes, boxes)], key=lambda x: x[0]))
            self._keyframes = [int(np.round(f)) for f in keyframes]  # coerce to int
            self._keyboxes = list(boxes)
        
    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.endframe() - self.startframe() > 0:
            strlist.append('startframe=%d, endframe=%d' % (self.startframe(), self.endframe()))
        strlist.append('keyframes=%d' % len(self._keyframes))
        return str('<vipy.object.track: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        """Interpolate the track at frame k"""
        return self._linear_interpolation(k)

    def __iter__(self):
        """Iterate over the track interpolating each frame from min(keyframes) to max(keyframes)"""
        for k in range(self.startframe(), self.endframe()+1):
            yield self._linear_interpolation(k)

    def __len__(self):
        """The length of a track is the total number of interpolated frames, or zero if degenerate"""
        return max(0, self.endframe() - self.startframe() + 1) if (len(self._keyframes)>0 and len(self._keyboxes)>0) else 0

    def isempty(self):
        return self.__len__() == 0

    def isdegenerate(self):
        return not (len(self.keyboxes()) == len(self.keyframes()) and
                    (len(self) == 0 or all([bb.isvalid() for bb in self.keyboxes()])) and
                    sorted(self.keyframes()) == list(self.keyframes()))
    
    def dict(self):
        return {'id':self._id, 'label':self.category(), 'shortlabel':self.shortlabel(), 'keyframes':self._keyframes, 'framerate':self._framerate, 
                'boundingbox':[bb.dict() for bb in self._keyboxes], 'attributes':self.attributes}

    def add(self, keyframe, box):
        """Add a new keyframe and associated box to track, preserve sorted order of keyframes"""
        assert isinstance(box, BoundingBox), "Invalid input - Box must be vipy.geometry.BoundingBox()"
        assert box.isvalid(), "Invalid input - Box must be non-degenerate"
        self._keyframes.append(keyframe)
        self._keyboxes.append(box)
        if len(self._keyframes) > 1 and keyframe < self._keyframes[-2]:
            (self._keyframes, self._keyboxes) = zip(*sorted([(f,bb) for (f,bb) in zip(self._keyframes, self._keyboxes)], key=lambda x: x[0]))        
            self._keyframes = list(self._keyframes)
            self._keyboxes = list(self._keyboxes)
        return self

    def update(self, keyframe, box):
        if keyframe in self._keyframes:
            self.delete(keyframe)
        self.add(keyframe, box)
        return self
        
    def replace(self, keyframe, box):
        """Replace a keyframe and associated box to track, preserve sorted order of keyframes"""
        assert isinstance(box, BoundingBox), "Invalid input - Box must be vipy.geometry.BoundingBox()"
        assert box.isvalid(), "Invalid input - Box must be non-degenerate"
        assert keyframe in self._keyframes, "Keyframe not found"
        self._keyboxes[self._keyframes.index(keyframe)] = box
        return self

    def delete(self, keyframe):
        """Replace a keyframe and associated box to track, preserve sorted order of keyframes"""
        assert keyframe in self._keyframes, "Keyframe not found"
        k = self._keyframes.index(keyframe)
        del self._keyboxes[k]
        del self._keyframes[k]
        if len(self._keyframes) == 0:
            warnings.warn('Empty track')
        return self
    
    def keyframes(self):
        """Return keyframe frame indexes where there are track observations"""
        return self._keyframes

    def keyboxes(self, boxes=None, keyframes=None):
        """Return keyboxes where there are track observations"""
        if boxes is None and keyframes is None:
            return self._keyboxes
        else:
            assert all([isinstance(bb, BoundingBox) for bb in boxes])
            self._keyboxes = boxes
            self._keyframes = keyframes if keyframes is not None else self._keyframes
            assert not self.isdegenerate()
            return self
        
    def meanshape(self):
        """Return the mean (width,height) of the box during the track"""
        return np.mean([bb.shape() for bb in self._keyboxes], axis=0)
            
    def framerate(self, fps):
        """Resample keyframes from known original framerate set by constructor to be new framerate fps"""
        assert self._framerate is not None, "Framerate conversion requires that the framerate is known for current keyframes.  This must be provided to the vipy.object.Track() constructor."
        self._keyframes = [int(np.round(f*(fps/float(self._framerate)))) for f in self._keyframes]
        self._framerate = fps
        return self
        
    def startframe(self):
        return np.min(self._keyframes) if len(self._keyframes)>0 else None

    def endframe(self):
        return np.max(self._keyframes) if len(self._keyframes)>0 else None

    def _linear_interpolation(self, k):
        """Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at keyframes.  
        This returns a vipy.object.Detection() which is the interpolation of the Track() at frame k
        If self._boundary='extend', then boxes are repeated if the interpolation is outside the keyframes
        If self._boundary='strict', then interpolation returns None if the interpolation is outside the keyframes
        """
        (xmin, ymin, width, height) = zip(*[bb.to_xywh() for bb in self._keyboxes])
        d = Detection(xmin=np.interp(k, self._keyframes, xmin),
                      ymin=np.interp(k, self._keyframes, ymin),
                      width=np.interp(k, self._keyframes, width),
                      height=np.interp(k, self._keyframes, height),
                      category=self.category(),
                      shortlabel=self.shortlabel())
        d.attributes['trackid'] = self.id()  # for correspondence of detections to tracks
        return d if self._boundary == 'extend' else (None if not self.during(k) else d)


    def category(self, label=None):
        if label is not None:
            self._label = label
            return self
        else:
            return self._label
    
    def label(self, label):
        """Alias for category"""
        return self.category(label)
        
    def shortlabel(self, label=None):
        """A optional shorter label string to show as a caption in visualizations"""                
        if label is not None:
            self._shortlabel = label
            return self
        else:
            return self._shortlabel

    def during(self, k_start, k_end=None):
        """Is frame during the time interval (startframe, endframe) inclusive?"""        
        k_end = k_start+1 if k_end is None else k_end
        return any([k >= self.startframe() and k <= self.endframe() for k in range(k_start, k_end)])

    def offset(self, dt=0, dx=0, dy=0):
        self._keyboxes = [bb.offset(dx, dy) for bb in self._keyboxes]
        self._keyframes = list(np.array(self._keyframes) + dt)
        return self

    def frameoffset(self, dx, dy):
        """Offset boxes by (dx,dy) in each frame"""
        assert len(self.keyboxes()) == len(dx) and len(self.keyboxes()) == len(dy)
        self._keyboxes = [bb.offset(dx=x, dy=y) for (bb, (x, y)) in zip(self._keyboxes, zip(dx, dy))]
        return self
        
    def rescale(self, s):
        """Rescale track boxes by scale factor s"""
        self._keyboxes = [bb.rescale(s) for bb in self._keyboxes]
        return self

    def scale(self, s):
        """Alias for rescale"""
        return self.rescale(s)

    def scalex(self, sx):
        """Rescale track boxes by scale factor sx"""
        self._keyboxes = [bb.scalex(sx) for bb in self._keyboxes]
        return self

    def scaley(self, sy):
        """Rescale track boxes by scale factor sx"""
        self._keyboxes = [bb.scaley(sy) for bb in self._keyboxes]
        return self

    def dilate(self, s):
        """Dilate track boxes by scale factor s"""
        self._keyboxes = [bb.dilate(s) for bb in self._keyboxes]
        return self

    def rot90cw(self, H, W):
        """Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent"""
        self._keyboxes = [bb.rot90cw(H, W) for bb in self._keyboxes]
        return self

    def rot90ccw(self, H, W):
        """Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent"""
        self._keyboxes = [bb.rot90ccw(H, W) for bb in self._keyboxes]
        return self

    def fliplr(self, H, W):
        """Flip an image left and right (mirror about vertical axis)"""
        self._keyboxes = [bb.fliplr(width=W) for bb in self._keyboxes]
        return self

    def flipud(self, H, W):
        """Flip an image left and right (mirror about vertical axis)"""
        self._keyboxes = [bb.flipud(height=H) for bb in self._keyboxes]
        return self

    def id(self, newid=None):
        if newid is None:
            return self._id
        else:
            self._id = newid
            return self

    def clone(self):
        return copy.deepcopy(self)

    def boundingbox(self):
        """The bounding box of a track is the smallest spatial box that contains all of the detections, or None if there are no detections"""
        d = self._keyboxes[0].clone() if len(self._keyboxes) >= 1 else None
        return d.union(self._keyboxes[1:]) if (d is not None and len(self._keyboxes) >= 2) else d

    def clip(self, startframe, endframe):
        """Clip a track to be within (startframe,endframe) with strict boundary handling"""
        if self[startframe] is not None:
            self.add(startframe, self[startframe])
        if self[endframe] is not None:
            self.add(endframe, self[endframe])
        keyframes = [f for (f,bb) in zip(self._keyframes, self._keyboxes) if f>=startframe and f<=endframe]  # may be empty
        keyboxes = [bb for (f,bb) in zip(self._keyframes, self._keyboxes) if f>=startframe and f<=endframe]  # may be empty
        if len(keyframes) == 0 or len(keyboxes) == 0:
            raise ValueError('Track does not contain any keyboxes within the requested frames (%d,%d)' % (startframe, endframe))
        self._keyframes = keyframes
        self._keyboxes = keyboxes
        self._boundary = 'strict'
        return self

    def iou(self, other, dt=1, n=None):
        """Compute the spatial IoU between two tracks as the mean IoU per frame in the range (self.startframe(), self.endframe())"""
        return self.rankiou(other, rank=len(self), dt=dt, n=n)

    def maxiou(self, other, dt=1, n=None):
        """Compute the maximum spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe())"""        
        return self.rankiou(other, rank=1, dt=dt, n=n)

    def endpointiou(self, other):
        """Compute the mean spatial IoU between two tracks at the two overlapping endpoints.  useful for track continuation"""        
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"
        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())
        return np.mean([self[startframe].iou(other[startframe]), self[endframe].iou(other[endframe])]) if endframe >= startframe else 0.0        

    def segmentiou(self, other, dt=5):
        """Compute the mean spatial IoU between two tracks at the overlapping segment, sampling by dt.  Useful for track continuation for densely overlapping tracks"""
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"
        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())   # inclusive
        return np.mean([self[min(k,endframe)].iou(other[min(k,endframe)]) for k in range(startframe, endframe, dt)]) if endframe >= startframe else 0.0 
        
    def rankiou(self, other, rank, dt=1, n=None):
        """Compute the mean spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe()) using only the top-k (rank) frame overlaps
           Sample tracks at endpoints and n uniformly spaced frames or a stride of dt frames.
        """
        assert rank >= 1 and rank <= len(self)
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        assert n is None or n >= 1
        assert dt >= 1
        dt = max(1, int(len(self)/n) if n is not None else dt)
        frames = [self.startframe()] + list(range(self.startframe()+dt, self.endframe(), dt)) + [self.endframe()]
        return np.mean(sorted([self[k].iou(other[k]) if (self.during(k) and other.during(k)) else 0.0 for k in frames])[-rank:])

    def percentileiou(self, other, percentile, dt=1, n=None):
        """Percentile iou returns rankiou for rank=percentile*len(self)"""
        assert percentile > 0 and percentile <= 1
        return self.rankiou(other, max(1, int(len(self)*percentile)), dt=dt, n=n)

    def average(self, other):
        """Compute the average of two tracks by the framewise interpolated boxes at the keyframes of this track"""
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        assert other.category() == self.category(), "Category mismatch"
        T = self.clone()
        T._keyboxes = [(self[k].average(other[k]) 
                        if (self.during(k) and other.during(k)) else 
                        (self[k] if self.during(k) and not other.during(k) else other[k]))
                       for k in T._keyframes]  
        return T  

    def smooth(self, width):
        """Track smoothing by averaging neighboring keyboxes"""
        assert isinstance(width, int)
        self._keyboxes = [bb.clone().average(bbnbrs) for (bb, bbnbrs) in zip(self._keyboxes, chunklistwithoverlap(self._keyboxes, width, width-1))]
        return self

    def smoothshape(self, width):
        """Track smoothing by averaging width and height of neighboring keyboxes"""
        assert isinstance(width, int)
        self._keyboxes = [bb.clone().averageshape(bbnbrs) for (bb, bbnbrs) in zip(self._keyboxes, chunklistwithoverlap(self._keyboxes, width, width-1))]
        return self

    def medianshape(self, width):
        """Track smoothing by median width and height of neighboring keyboxes"""
        assert isinstance(width, int)
        self._keyboxes = [bb.clone().medianshape(bbnbrs) for (bb, bbnbrs) in zip(self._keyboxes, chunklistwithoverlap(self._keyboxes, width, width-1))]
        return self

    def spline(self, smoothingfactor=None):
        """Track smoothing by cubic spline fit, will return resampled dt=1 track.  Smoothing factor will increase with smoothing > 1 and decrease with 0 < smoothing < 1"""
        assert smoothingfactor is None or smoothingfactor > 0
        t = self.clone().resample(dt=1)
        s = smoothingfactor * len(self._keyframes) if smoothingfactor is not None else None
        (xmin, ymin, xmax, ymax) = zip(*[bb.to_ulbr() for bb in t._keyboxes])
        f_xmin = scipy.interpolate.UnivariateSpline(t._keyframes, xmin, check_finite=False, s=s)
        f_ymin = scipy.interpolate.UnivariateSpline(t._keyframes, ymin, check_finite=False, s=s)
        f_xmax = scipy.interpolate.UnivariateSpline(t._keyframes, xmax, check_finite=False, s=s)
        f_ymax = scipy.interpolate.UnivariateSpline(t._keyframes, ymax, check_finite=False, s=s)
        (self._keyframes, self._keyboxes) = zip(*[(k, BoundingBox(xmin=float(f_xmin(k)), ymin=float(f_ymin(k)), xmax=float(f_xmax(k)), ymax=float(f_ymax(k)))) for k in range(self.startframe(), self.endframe())])
        return self

    def imclip(self, width, height):
        """Clip the track to the image rectangle (width, height).  If a keybox is outside the image rectangle, remove it otherwise clip to the image rectangle. 
           This operation can change the length of the track and the size of the keyboxes.  The result may be an empty track if the track is completely outside
           the image rectangle, which results in an exception.
        """
        clipped = [(f, bb.imclip(width=width, height=height)) for (f,bb) in zip(self._keyframes, self._keyboxes) if bb.hasoverlap(width=width, height=height)]
        if len(clipped) > 0:
            (self._keyframes, self._keyboxes) = zip(*clipped)
            (self._keyframes, self._keyboxes) = (list(self._keyframes), list(self._keyboxes))
            return self
        else:
            raise ValueError('All key boxes for track outside image rectangle')

    def resample(self, dt):
        """Resample the track using a stride of dt frames"""
        assert dt >= 1 and dt < len(self)
        frames =  list(range(self.startframe(), self.endframe(), dt)) + [self.endframe()]
        (self._keyboxes, self._keyframes) = zip(*[(self[k], k) for k in frames])
        (self._keyboxes, self._keyframes) = (list(self._keyboxes), list(self._keyframes))
        return self


def non_maximum_suppression(detlist, conf, iou, bycategory=False):
    """Compute non-maximum suppression of a list of vipy.object.Detection() based on spatial IOU threshold (iou) and a confidence threshold (conf)"""
    assert all([isinstance(d, Detection) for d in detlist])
    assert all([d.confidence() is not None for d in detlist])
    assert conf>=0 and iou>=0 and iou<=1

    detlist = [d for d in detlist if d.confidence() > conf and not d.isdegenerate()]    
    detlist = sorted(detlist, key=lambda d: d.confidence(), reverse=True)  # biggest to smallest
    
    suppresslist = []
    for (i,di) in enumerate(detlist):
        for (j,dj) in enumerate(detlist):
            if j > i and di.iou(dj) > iou and (bycategory is False or di.category() == dj.category()):
                suppresslist.append(j)
    return sorted([d for (j,d) in enumerate(detlist) if j not in set(suppresslist)], key=lambda d: d.confidence())  # smallest to biggest for display layering


def greedy_assignment(srclist, dstlist, miniou=0.0):
    """Compute a greedy one-to-one assignment of each vipy.object.Detection() in srclist to a unique element in dstlist with the largest IoU greater than miniou, else None
    
       returns:
          assignlist [list]:  [d.iou(dstlist[j]) if j is not None else 0 for (d,j) in zip(srclist, assignlist)]  is the IoU for the assignment
    """
    assert all([isinstance(d, Detection) for d in srclist])
    assert all([isinstance(d, Detection) for d in dstlist])    
    
    assignlist = []
    for ds in sorted(srclist, key=lambda d: d.area(), reverse=True):
        iou = [ds.iou(d) if j not in assignlist else 0.0 for (j,d) in enumerate(dstlist)]
        assignlist.append( np.argmax(iou) if max(iou) > miniou else None)
    return assignlist
