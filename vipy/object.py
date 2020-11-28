import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist, chunklistwithoverlap, try_import, Timer
import uuid
import copy
import warnings

try:
    import ujson as json  # faster
except ImportError:
    import json


class Detection(BoundingBox):
    """vipy.object.Detection class
    
    This class represent a single object detection in the form a bounding box with a label and confidence.
    The constructor of this class follows a subset of the constructor patterns of vipy.geometry.BoundingBox

    >>> d = vipy.object.Detection(category='Person', xmin=0, ymin=0, width=50, height=100)
    >>> d = vipy.object.Detection(label='Person', xmin=0, ymin=0, width=50, height=100)  # "label" is an alias for "category"
    >>> d = vipy.object.Detection(label='John Doe', shortlabel='Person', xmin=0, ymin=0, width=50, height=100)  # shortlabel is displayed
    >>> d = vipy.object.Detection(label='Person', xywh=[0,0,50,100])
    >>> d = vupy.object.Detection(..., id=True)  # generate a unique UUID for this detection retrievable with d.id()

    """

    def __init__(self, label=None, xmin=None, ymin=None, width=None, height=None, xmax=None, ymax=None, confidence=None, xcentroid=None, ycentroid=None, category=None, xywh=None, shortlabel=None, attributes=None, id=True):
        super().__init__(xmin=xmin, ymin=ymin, width=width, height=height, xmax=xmax, ymax=ymax, xcentroid=xcentroid, ycentroid=ycentroid, xywh=xywh)
        assert not (label is not None and category is not None), "Constructor requires either label or category kwargs, not both"
        self._id = uuid.uuid4().hex if id is True else (None if id is False else id)  # unique id if id=True
        self._label = category if category is not None else label
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._confidence = float(confidence) if confidence is not None else confidence
        self.attributes = {} if attributes is None else attributes

    @classmethod
    def cast(cls, d, flush=False):
        assert isinstance(d, BoundingBox)
        if d.__class__ != Detection:
            d.__class__ = Detection
            d._id = uuid.uuid4().hex if flush or not hasattr(d, '_id') else d._id
            d._shortlabel = None if flush or not hasattr(d, '_shortlabel') else d._shortlabel
            d._confidence = None if flush or not hasattr(d, '_confidence') else d._confidence
            d._label = None if flush or not hasattr(d, '_label') else d._label
            d.attributes = {} if flush or not hasattr(d, 'attributes') else d.attributes
        return d
        
    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s        
        return obj(xmin=d['_xmin'], ymin=d['_ymin'], xmax=d['_xmax'], ymax=d['_ymax'], label=d['_label'], shortlabel=d['_shortlabel'], confidence=d['_confidence'], attributes=d['attributes'])
        
    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if True:
            strlist.append('bbox=(xmin=%1.1f, ymin=%1.1f, width=%1.1f, height=%1.1f)' %
                           (self.xmin(), self.ymin(),self.width(), self.height()))
        if self._confidence is not None:
            strlist.append('conf=%1.3f' % self.confidence())
        if self.isdegenerate():
            strlist.append('degenerate')
        return str('<vipy.object.detection: %s>' % (', '.join(strlist)))

    def __eq__(self, other):
        """Detection equality when bounding boxes and categories are equivalent"""
        return isinstance(other, Detection) and self.xywh() == other.xywh() and self.category() == other.category()

    def __str__(self):
        return self.__repr__()

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(s=None, encode=False)

    def json(self, encode=True):
        return json.dumps(self.__dict__) if encode else self.__dict__
                
    def nocategory(self):
        self._label = None
        return self

    def noshortlabel(self):
        self._shortlabel = None
        return self
        
    def category(self, category=None, shortlabel=True):
        """Update the category and shortlabel (optional) of the detection"""
        if category is None:
            return self._label
        else:
            self._label = str(category)  # coerce to string
            self._shortlabel = str(category) if shortlabel else self._shortlabel  # coerce to string            
            return self

    def shortlabel(self, label=None):
        """A optional shorter label string to show in the visualizations, defaults to category()"""        
        if label is not None:
            self._shortlabel = str(label)  # coerce to string
            return self
        else:
            return self._shortlabel if self._shortlabel is not None else self.category()

    def label(self, label):
        """Alias for category to update both category and shortlabel"""
        return self.category(label, shortlabel=True)

    def id(self):
        return self._id

    def clone(self):
        return copy.deepcopy(self)

    def confidence(self, c=None):
        if c is None:
            return self._confidence
        else:
            self._confidence = c
            return self

    def hasattribute(self, k):
        return isinstance(self.attributes, dict) and k in self.attributes

    
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
                
        self._id = uuid.uuid4().hex if trackid is None else trackid
        self._label = category if category is not None else label
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._framerate = framerate
        self._interpolation = interpolation
        self._boundary = self.boundary(boundary)
        self.attributes = attributes if attributes is not None else {}        
        self._keyframes = [int(np.round(f)) for f in keyframes]  # coerce to int
        self._keyboxes = boxes
        
        # Sorted increasing frame order
        if len(keyframes) > 0 and len(boxes) > 0 and not all([keyframes[i-1] <= keyframes[i] for i in range(1,len(keyframes))]):
            (keyframes, boxes) = zip(*sorted([(f,bb) for (f,bb) in zip(keyframes, boxes)], key=lambda x: x[0]))
            self._keyframes = list(keyframes)
            self._keyboxes = list(boxes)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        return cls(keyframes=[int(f) for f in d['_keyframes']],
                   boxes=[BoundingBox.from_json(bbs) for bbs in d['_keyboxes']],
                   category=d['_label'],
                   confidence=None,
                   framerate=d['_framerate'],
                   interpolation=d['_interpolation'],
                   boundary=d['_boundary'],
                   shortlabel=d['_shortlabel'],
                   attributes=d['attributes'],
                   trackid=d['_id'])
                   
    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.endframe() is not None and self.startframe() is not None and ((self.endframe() - self.startframe()) > 0):
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
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(encode=False)

    def json(self, encode=True):
        d = {k:v if k != '_keyboxes' else [bb.json(encode=False) for bb in v] for (k,v) in self.__dict__.items()}
        d['_keyframes'] = [int(f) for f in self._keyframes]
        return json.dumps(d) if encode else d
    
    def add(self, keyframe, bbox, strict=True):
        """Add a new keyframe and associated box to track, preserve sorted order of keyframes. 

           -strict [bool]:  If box is degenerate, throw an exception if strict=True, otherwise just don't add it
        """
        assert isinstance(bbox, BoundingBox), "Invalid input - Box must be vipy.geometry.BoundingBox()"
        assert strict is False or bbox.isvalid(), "Invalid input - Box must be non-degenerate"
        if not bbox.isvalid():
            return self  # just don't add it 
        self._keyframes.append(int(keyframe))
        self._keyboxes.append(bbox)
        if len(self._keyframes) > 1 and keyframe < self._keyframes[-2]:
            (self._keyframes, self._keyboxes) = zip(*sorted([(f,bb) for (f,bb) in zip(self._keyframes, self._keyboxes)], key=lambda x: x[0]))        
            self._keyframes = list(self._keyframes)
            self._keyboxes = list(self._keyboxes)
        return self

    def update(self, keyframe, bbox):
        if keyframe in self._keyframes:
            self.delete(keyframe)
        self.add(keyframe, bbox)
        return self
        
    def replace(self, keyframe, box):
        """Replace the keyframe and associated box(es), preserve sorted order of keyframes"""
        return self.delete(keyframe).add(keyframe, box)

    def delete(self, keyframe):
        """Replace a keyframe and associated box to track, preserve sorted order of keyframes"""
        while keyframe in self._keyframes:
            k = self._keyframes.index(keyframe)
            del self._keyboxes[k]
            del self._keyframes[k]
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
        s = np.mean([bb.shape() for bb in self._keyboxes], axis=0)
        return (float(s[0]), float(s[1]))
            
    def framerate(self, fps=None, speed=None):
        """Resample keyframes from known original framerate set by constructor to be new framerate fps"""
        assert self._framerate is not None, "Framerate conversion requires that the framerate is known for current keyframes.  This must be provided to the vipy.object.Track() constructor."
        assert fps is not None or speed is not None, "Invalid input"
        assert not (fps is not None and speed is not None), "Invalid input"
        assert speed is None or speed > 0, "Invalid speed, must specify speed multiplier s=1, s=2 for 2x faster, s=0.5 for half slower"
        
        fps = fps if fps is not None else (1.0/speed)*self._framerate
        self._keyframes = [int(np.round(f*(fps/float(self._framerate)))) for f in self._keyframes]
        self._framerate = fps
        return self
        
    def startframe(self):
        return int(min(self._keyframes)) if len(self._keyframes)>0 else None

    def endframe(self):
        return int(max(self._keyframes)) if len(self._keyframes)>0 else None

    def _linear_interpolation(self, k):
        """Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at keyframes.  
        This returns a vipy.object.Detection() which is the interpolation of the Track() at frame k
        If self._boundary='extend', then boxes are repeated if the interpolation is outside the keyframes
        If self._boundary='strict', then interpolation returns None if the interpolation is outside the keyframes
        """
        assert not self.isempty(), "Degenerate object for interpolation"
        (xmin, ymin, width, height) = zip(*[bb.to_xywh() for bb in self._keyboxes])
        d = Detection(xmin=float(np.interp(k, self._keyframes, xmin)),
                      ymin=float(np.interp(k, self._keyframes, ymin)),
                      width=float(np.interp(k, self._keyframes, width)),
                      height=float(np.interp(k, self._keyframes, height)),
                      category=self.category(),
                      shortlabel=self.shortlabel())
        d.attributes['trackid'] = self.id()  # for correspondence of detections to tracks
        return d if self._boundary == 'extend' or self.during(k) else None


    def category(self, label=None, shortlabel=True):
        if label is not None:
            self._label = str(label)  # coerce to string
            self._shortlabel = str(label) if shortlabel else self._shortlabel  # coerce to string
            return self
        else:
            return self._label
    
    def label(self, label):
        """Alias for category"""
        return self.category(label, shortlabel=True)
        
    def shortlabel(self, label=None):
        """A optional shorter label string to show as a caption in visualizations"""                
        if label is not None:
            self._shortlabel = str(label)  # coerce to string
            return self
        else:
            return self._shortlabel

    def during(self, k_start, k_end=None):
        """Is frame during the time interval (startframe, endframe) inclusive?"""        
        k_end = k_start+1 if k_end is None else k_end
        return len(self)>0 and any([k >= self.startframe() and k <= self.endframe() for k in range(k_start, k_end)])

    def offset(self, dt=0, dx=0, dy=0):
        self._keyboxes = [bb.offset(dx, dy) for bb in self._keyboxes]
        self._keyframes = [int(f+dt) for f in self._keyframes]
        return self

    def frameoffset(self, dx, dy):
        """Offset boxes by (dx,dy) in each frame"""
        assert len(self.keyboxes()) == len(dx) and len(self.keyboxes()) == len(dy)
        self._keyboxes = [bb.offset(dx=x, dy=y) for (bb, (x, y)) in zip(self._keyboxes, zip(dx, dy))]
        return self

    def truncate(self, startframe=None, endframe=None):
        """Truncate a track so that any keyframes less than startframe or greater than or equal to endframe are removed"""
        keyframes = copy.deepcopy(self.keyframes())
        for k in keyframes:
            if ((startframe is not None and k < startframe) or (endframe is not None and k >= endframe)):
                self.delete(k)
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
        """The bounding box of a track is the smallest spatial box that contains all of the detections within startframe and endframe, or None if there are no detections"""
        d = self._keyboxes[0].clone() if len(self._keyboxes) >= 1 else None
        return d.union([bb for (k,bb) in zip(self._keyframes[1:], self._keyboxes[1:]) if self.during(k)]) if (d is not None and len(self._keyboxes) >= 2) else d

    def pathlength(self):
        """The path length of a track is the cumulative Euclidean distance in pixels that the box travels"""
        return float(np.sum([bb_next.dist(bb_prev) for (bb_next, bb_prev) in zip(self._keyboxes[1:], self._keyboxes[0:-1])])) if len(self._keyboxes)>1 else 0.0
        
    def boundary(self, b=None):
        if b is None:
            return self._boundary
        else:
            assert b in ['strict', 'extend']
            self._boundary = b
            return self
        
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
        return float(np.mean([self[startframe].iou(other[startframe]), self[endframe].iou(other[endframe])]) if endframe > startframe else 0.0)

    def segmentiou(self, other, dt=5):
        """Compute the mean spatial IoU between two tracks at the overlapping segment, sampling by dt.  Useful for track continuation for densely overlapping tracks"""
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"
        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())   # inclusive
        return float(np.mean([self[min(k,endframe)].iou(other[min(k,endframe)]) for k in range(startframe, endframe, dt)]) if endframe > startframe else 0.0)
        
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
        return float(np.mean(sorted([self[k].iou(other[k]) if (self.during(k) and other.during(k)) else 0.0 for k in frames])[-rank:]))

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

    def spline(self, smoothingfactor=None, strict=True, startframe=None, endframe=None):
        """Track smoothing by cubic spline fit, will return resampled dt=1 track.  Smoothing factor will increase with smoothing > 1 and decrease with 0 < smoothing < 1
        
           This function requires optional package scipy
        """
        try_import('scipy', 'scipy');  import scipy.interpolate;
        assert smoothingfactor is None or smoothingfactor > 0
        t = self.clone().resample(dt=1)
        (startframe, endframe) = (self.startframe() if startframe is None else startframe, self.endframe() if endframe is None else endframe)
        try:
            assert len(t._keyframes) > 4, "Invalid length for spline interpolation"        
            s = smoothingfactor * len(self._keyframes) if smoothingfactor is not None else None
            (xmin, ymin, xmax, ymax) = zip(*[bb.to_ulbr() for bb in t._keyboxes])
            f_xmin = scipy.interpolate.UnivariateSpline(t._keyframes, xmin, check_finite=False, s=s)
            f_ymin = scipy.interpolate.UnivariateSpline(t._keyframes, ymin, check_finite=False, s=s)
            f_xmax = scipy.interpolate.UnivariateSpline(t._keyframes, xmax, check_finite=False, s=s)
            f_ymax = scipy.interpolate.UnivariateSpline(t._keyframes, ymax, check_finite=False, s=s)
            (self._keyframes, self._keyboxes) = zip(*[(k, BoundingBox(xmin=float(f_xmin(k)), ymin=float(f_ymin(k)), xmax=float(f_xmax(k)), ymax=float(f_ymax(k)))) for k in range(startframe, endframe)])
        except Exception as e:
            if not strict:
                print('[vipy.object.track]: spline smoothing failed with error "%s" - Returning unsmoothed track' % (str(e)))
                return self
            else:
                raise
        return self

    def linear_extrapolation(self, k, shape=False):
        """Track extrapolation by linear fit.
        
           * Requires at least 2 keyboxes.
           * Returned boxes may be degenerate.
           * shape=True then both the position and shape (width, height) of the box is extrapolated
        """
        if self.during(k):
            return self[k]
        elif len(self.keyboxes()) == 1:
            return self.nearest_keybox(k)
        else:
            n = self.endframe() if k > self.endframe() else self.startframe()+1
            (vx, vy, vw, vh) = (self.velocity_x(n, dt=1), self.velocity_y(n, dt=1), self.velocity_w(n, dt=1), self.velocity_h(n, dt=1))
            return (self.clone()[n]
                    .translate((k-n)*vx, (k-n)*vy)
                    .top(0 if not shape else ((k-n)*vh)/2.0)
                    .bottom(0 if not shape else ((k-n)*vh)/2.0)
                    .left(0 if not shape else ((k-n)*vw)/2.0)
                    .right(0 if not shape else ((k-n)*vw)/2.0))
       
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
        """Resample the track using a stride of dt frames.  This reduces the density of keyframes by interpolating new keyframes as a uniform stride of dt.  This is useful for track compression"""
        assert dt >= 1 and dt < len(self)
        frames =  list(range(self.startframe(), self.endframe(), dt)) + [self.endframe()]
        (self._keyboxes, self._keyframes) = zip(*[(self[k], k) for k in frames])
        (self._keyboxes, self._keyframes) = (list(self._keyboxes), list(self._keyframes))
        return self

    def significant_digits(self, n):
        self._keyboxes = [bb.significant_digits(n) for bb in self._keyboxes]
        return self

    def velocity(self, f, dt=5):
        """Return the (x,y) track velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)              
        return (self.velocity_x(f, dt), self.velocity_y(f, dt))

    def velocity_x(self, f, dt=5):
        """Return the (x) track velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return (self[f].centroid_x() - self[f-dt].centroid_x())/float(dt)

    def velocity_y(self, f, dt=5):
        """Return the (y) track velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return (self[f].centroid_y() - self[f-dt].centroid_y())/float(dt)

    def velocity_w(self, f, dt=5):
        """Return the (w) track velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return (self[f].width() - self[f-dt].width())/float(dt)

    def velocity_h(self, f, dt=5):
        """Return the (h) track velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return (self[f].height() - self[f-dt].height())/float(dt)
    
    def nearest_keyframe(self, f):
        """Nearest keyframe to frame f"""
        return self._keyframes[int(np.abs(np.array(self._keyframes) - f).argmin())]

    def nearest_keybox(self, f):
        """Nearest keybox to frame f"""
        return self._keyboxes[int(np.abs(np.array(self._keyframes) - f).argmin())]
    
    
def non_maximum_suppression(detlist, conf, iou, bycategory=False, cover=None):
    """Compute greedy non-maximum suppression of a list of vipy.object.Detection() based on spatial IOU threshold (iou) and cover threhsold (cover) sorted by confidence (conf)"""
    assert all([isinstance(d, Detection) for d in detlist])
    assert all([d.confidence() is not None for d in detlist])
    assert conf>=0 and iou>=0 and iou<=1
    assert cover is None or (cover>=0 and cover<=1)

    suppressed = set([k for (k,d) in enumerate(detlist) if d.confidence() <= conf or d.isdegenerate()])
    detlist = sorted(detlist, key=lambda d: d.confidence(), reverse=True)  # biggest to smallest
    for (i,di) in enumerate(detlist):
        for (j,dj) in enumerate(detlist):
            if j > i and (j not in suppressed) and (bycategory is False or di.category() == dj.category()) and (di.iou(dj) >= iou or (cover is not None and dj.cover(di) >= cover)):
                suppressed.add(j)
    return sorted([d for (j,d) in enumerate(detlist) if j not in suppressed], key=lambda d: d.confidence())  # smallest to biggest confidence for display layering


def greedy_assignment(srclist, dstlist, miniou=0.0, bycategory=False):
    """Compute a greedy one-to-one assignment of each vipy.object.Detection() in srclist to a unique element in dstlist with the largest IoU greater than miniou, else None
    
       returns:
          assignlist [list]:  same length as srclist, where j=assignlist[i] is the index of the assignment such that srclist[i] == dstlist[j]
    """
    assert all([isinstance(d, Detection) for d in srclist])
    assert all([isinstance(d, Detection) for d in dstlist])    
    assert miniou >= 0 and miniou <= 1.0
    
    assigndict = {}
    for (k, ds) in sorted(enumerate(srclist), key=lambda x: x[1].area(), reverse=True):
        iou = [ds.iou(d) if (j not in assigndict.values() and (bycategory is False or ds.category() == d.category())) else 0.0 for (j,d) in enumerate(dstlist)]
        assigndict[k] = np.argmax(iou) if len(iou) > 0 and max(iou) > miniou else None
    return [assigndict[k] for k in range(0, len(srclist))]
