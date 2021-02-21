import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist, chunklistwithoverlap, try_import, Timer
import uuid
import copy
import warnings
from itertools import islice

try:
    import ujson as json  # faster
except ImportError:
    import json

DETECTION_GUID = int(uuid.uuid4().hex[0:8], 16)  

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
        if id is True:
            global DETECTION_GUID; self._id = hex(int(DETECTION_GUID))[2:];  DETECTION_GUID = DETECTION_GUID + 1;  # faster, increment package level UUID4 initialized GUID
        else:
            self._id = None if id is False else id
        self._label = category if category is not None else label
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._confidence = float(confidence) if confidence is not None else confidence
        self.attributes = {} if attributes is None else attributes.copy()  # shallow copy

    @classmethod
    def cast(cls, d, flush=False, category=None):
        assert isinstance(d, BoundingBox)
        if d.__class__ != Detection:
            d.__class__ = Detection
            global DETECTION_GUID; newid = hex(int(DETECTION_GUID))[2:];  DETECTION_GUID = DETECTION_GUID + 1;  
            d._id = newid if flush or not hasattr(d, '_id') else d._id
            d._shortlabel = None if flush or not hasattr(d, '_shortlabel') else d._shortlabel
            d._confidence = None if flush or not hasattr(d, '_confidence') else d._confidence
            d._label = None if flush or not hasattr(d, '_label') else d._label
            d.attributes = {} if flush or not hasattr(d, 'attributes') else d.attributes
            if category is not None:
                d._label = category  # when casting BoundingBox to Detection, extra args are necessary
        return d
        
    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s        
        return obj(xmin=d['_xmin'], ymin=d['_ymin'], xmax=d['_xmax'], ymax=d['_ymax'],
                   label=d['_label'] if '_label' in d else None,
                   shortlabel=d['_shortlabel'] if '_shortlabel' in d else None,
                   confidence=d['_confidence'] if '_confidence' in d else None,
                   attributes=d['attributes'] if 'attributes' in d else None)
        
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
        return ((isinstance(other, Detection) and self.xywh() == other.xywh() and self.category() == other.category()) or
                isinstance(other, BoundingBox) and self.xywh() == other.xywh())

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
        #return copy.deepcopy(self)
        return Detection.from_json(self.json(encode=False))

    def confidence(self, c=None):
        if c is None:
            return self._confidence
        else:
            self._confidence = c
            return self

    def hasattribute(self, k):
        return isinstance(self.attributes, dict) and k in self.attributes

    def setattribute(self, k, v):
        self.attributes[k] = v
        return self
    
    def delattribute(self, k):
        self.attributes.pop(k, None)
        return self

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

    def __init__(self, keyframes, boxes, category=None, label=None, framerate=None, interpolation='linear', boundary='strict', shortlabel=None, attributes=None, trackid=None):

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
        self._boundary = boundary
        self.attributes = attributes.copy() if attributes is not None else {}  # shallow copy
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
        return cls(keyframes=tuple(int(f) for f in d['_keyframes']),
                   boxes=tuple([Detection.from_json(bbs) for bbs in d['_keyboxes']]),
                   category=d['_label'] if '_label' in d else None,
                   framerate=d['_framerate'],
                   interpolation=d['_interpolation'],
                   boundary=d['_boundary'],
                   shortlabel=d['_shortlabel'] if '_shortlabel' in d else None,
                   attributes=d['attributes'],
                   trackid=d['_id'])
                   
    def json(self, encode=True):
        d = {k:v if k != '_keyboxes' else tuple([bb.json(encode=False) for bb in v]) for (k,v) in self.__dict__.items()}
        d['_keyframes'] = tuple([int(f) for f in self._keyframes])
        return json.dumps(d) if encode else d

    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        if self.endframe() is not None and self.startframe() is not None:
            strlist.append('startframe=%d, endframe=%d' % (self.startframe(), self.endframe()))
        strlist.append('keyframes=%d' % len(self._keyframes))
        return str('<vipy.object.track: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        """Interpolate the track at frame k"""
        return self.linear_interpolation(k)

    def __iter__(self):
        """Iterate over the track interpolating each frame from min(keyframes) to max(keyframes)"""
        for k in range(self.startframe(), self.endframe()+1):
            yield self.linear_interpolation(k)

    def __len__(self):
        """The length of a track is the total number of interpolated frames, or zero if degenerate"""
        return max(0, self.endframe() - self.startframe() + 1) if (len(self._keyframes)>0 and len(self._keyboxes)>0) else 0

    def isempty(self):
        return self.__len__() == 0

    def confidence(self, last=None, samples=None):
        """The confidence of a track is the mean confidence of all (or just last=last frames, or samples=samples uniformly spaced) keyboxes (if confidences are available) else 0"""
        if samples is not None:
            dt = max(1, int(round(len(self._keyframes)/float(samples))))
            C = [self._keyboxes[i]._confidence for i in range(len(self._keyframes)-1, -1, -dt) if (hasattr(self._keyboxes[i], '_confidence') and self._keyboxes[i]._confidence is not None)]
        elif last == 1:
            return self.endbox().confidence() if len(self)>0 else 0
        else:
            ef = self.endframe() - last if last is not None else 0
            C = [d._confidence for (f,d) in zip(self.keyframes(), self.keyboxes()) if f >= ef and (hasattr(d, '_confidence') and d._confidence is not None)]
        return C[0] if len(C) == 1 else (float(np.mean(C)) if len(C) > 0 else 0)
        
    def isdegenerate(self):
        return not (len(self.keyboxes()) == len(self.keyframes()) and
                    (len(self) == 0 or all([bb.isvalid() for bb in self.keyboxes()])) and
                    sorted(self.keyframes()) == list(self.keyframes()))
    
    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(encode=False)

    
    def add(self, keyframe, bbox, strict=True):
        """Add a new keyframe and associated box to track, preserve sorted order of keyframes. 

           -strict [bool]:  If box is degenerate, throw an exception if strict=True, otherwise just don't add it
        """
        assert isinstance(bbox, BoundingBox), "Invalid input - Box must be vipy.geometry.BoundingBox()"
        assert strict is False or bbox.isvalid(), "Invalid input - Box must be non-degenerate"
        assert int(keyframe) not in self._keyframes, "Invalid input - repeated keyframe"
        if not bbox.isvalid():            
            return self  # just don't add it 
        self._keyframes.append(int(keyframe))
        self._keyboxes.append(bbox)
        if len(self._keyframes) > 1 and keyframe < self._keyframes[-2]:
            # Preserve sorted order if inserting into the middle somewhere
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

    def num_keyframes(self):
        return len(self._keyframes)

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
        """Return the mean (width,height) of the box during the track, or None if the track is degenerate"""
        s = np.mean([bb.shape() for bb in self.keyboxes()], axis=0) if len(self.keyboxes()) > 0 else None
        return (float(s[0]), float(s[1])) if s is not None else None

    def meanbox(self):
        """Return the mean bounding box during the track, or None if the track is degenerate"""
        return BoundingBox(ulbr=np.mean([bb.ulbr() for bb in self.keyboxes()], axis=0)) if len(self.keyboxes()) > 0 else None 
    
    def shapevariance(self):
        """Return the variance (width, height) of the box shape during the track or None if the track is degenerate.  This is useful for filtering spurious tracks where the aspect ratio changes rapidly and randomly"""
        m = self.meanshape()
        return (float(np.mean([(bb.width() - m[0])**2 for bb in self.keyboxes()])), 
                float(np.mean([(bb.height() - m[1])**2 for bb in self.keyboxes()]))) if m is not None else None

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
        return self._keyframes[0] if len(self._keyframes)>0 else None  # assumes sorted order

    def endframe(self):
        return self._keyframes[-1] if len(self._keyframes)>0 else None  # assumes sorted order

    def linear_interpolation(self, f, id=True):
        """Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at keyframes.  
        This returns a vipy.object.Detection() which is the interpolation of the Track() at frame k
        If self._boundary='extend', then boxes are repeated if the interpolation is outside the keyframes
        If self._boundary='strict', then interpolation returns None if the interpolation is outside the keyframes
        
          - The returned object is not cloned when possible, be careful when modifying this object. 

        """
        assert len(self._keyboxes) > 0, "Degenerate object for interpolation"   # not self.isempty()
        if len(self._keyboxes) == 1:
            return Detection.cast(self._keyboxes[0].clone(), category=self.category()).setattribute('trackid', self.id()) if (self._boundary == 'extend' or self.during(f)) else None
        if f in reversed(self._keyframes):            
            return Detection.cast(self._keyboxes[self._keyframes.index(f)], category=self.category()).setattribute('trackid', self.id())

        kf = self._keyframes
        ft = min(max(f, kf[0]), kf[-1])  # truncated frame index
        for i in reversed(range(0, len(kf)-1)):
            if kf[i] <= ft and kf[i+1] >= ft:
                break  # floor keyframe index
        c = (ft - kf[i]) / max(1, float(kf[i+1] - kf[i]))  # interpolation coefficient
        (bi, bj) = (self._keyboxes[i], self._keyboxes[i+1])
        d = Detection(xmin=bi._xmin + c*(bj._xmin - bi._xmin),   # float(np.interp(k, self._keyframes, [bb._xmin for bb in self._keyboxes])),
                      ymin=bi._ymin + c*(bj._ymin - bi._ymin),   # float(np.interp(k, self._keyframes, [bb._ymin for bb in self._keyboxes])),
                      xmax=bi._xmax + c*(bj._xmax - bi._xmax),   # float(np.interp(k, self._keyframes, [bb._xmax for bb in self._keyboxes])),
                      ymax=bi._ymax + c*(bj._ymax - bi._ymax),   # float(np.interp(k, self._keyframes, [bb._ymax for bb in self._keyboxes])),
                      confidence=bi.confidence(),  # may be None
                      category=self.category(),
                      shortlabel=self.shortlabel(),
                      id=id)
        d.attributes['trackid'] = self.id()  # for correspondence of detections to tracks
        return d if self._boundary == 'extend' or self.during(f) else None

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
        (startframe, endframe) = (self.startframe(), self.endframe())
        return len(self)>0 and ((k_start >= startframe and k_start <= endframe) or (k_end >= startframe and k_end <= endframe) or (k_start <= startframe and k_end >= endframe))
        
    def during_interval(self, k_start, k_end):
        return self.during(k_start, k_end)

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
        """Truncate a track so that any keyframes less than startframe or greater than endframe are removed"""
        kfkb = [(kf,kb) for (kf,kb) in zip(self._keyframes, self._keyboxes) if ((startframe is None or kf >= startframe) and (endframe is None or kf <= endframe))]
        (self._keyframes, self._keyboxes) = zip(*kfkb) if len(kfkb) > 0 else ([], [])
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

    def clone(self, startframe=None, endframe=None):
        #return copy.deepcopy(self)  
        return Track.from_json(self.json(encode=False)) if (startframe is None and endframe is None) else self.clone_during(startframe, endframe)  # 2x faster than deepcopy

    def clone_during(self, startframe, endframe):
        #return copy.deepcopy(self)
        kfkb = [(kf,kb.clone()) for (kf,kb) in zip(self._keyframes, self._keyboxes) if ((startframe is None or kf >= startframe) and (endframe is None or kf <= endframe))]
        (kf, kb) = zip(*kfkb) if len(kfkb) > 0 else ([], [])        
        return Track(keyframes=kf, boxes=kb, category=self._label, framerate=self._framerate, interpolation=self._interpolation, boundary=self._boundary, shortlabel=self._shortlabel, attributes=self.attributes, trackid=self._id)
    
    def boundingbox(self, startframe=None, endframe=None):
        """The bounding box of a track is the smallest spatial box that contains all of the detections within startframe and endframe, or None if there are no detections"""
        t = self.clone() if (startframe is None and endframe is None) else self.clone().truncate(startframe, endframe)
        d = t._keyboxes[0].clone() if len(t._keyboxes) >= 1 else None
        return d.union([bb for (k,bb) in zip(t._keyframes[1:], t._keyboxes[1:]) if t.during(k)]) if (d is not None and len(t._keyboxes) >= 2) else d

    def smallestbox(self):
        """The smallest box of a track is the smallest spatial box in area along the track"""
        k = np.argmin([bb.area() for bb in self._keyboxes]) if len(self._keyboxes) > 0 else None
        return self._keyboxes[k] if k is not None else None

    def biggestbox(self):
        """The biggest box of a track is the largest spatial box in area along the track"""
        k = np.argmax([bb.area() for bb in self._keyboxes]) if len(self._keyboxes) > 0 else None
        return self._keyboxes[k] if k is not None else None
        
    def pathlength(self):
        """The path length of a track is the cumulative Euclidean distance in pixels that the box travels"""
        return float(np.sum([bb_next.dist(bb_prev) for (bb_next, bb_prev) in zip(self._keyboxes[1:], self._keyboxes[0:-1])])) if len(self._keyboxes)>1 else 0.0
        
    def startbox(self):
        """The startbox is the first bounding box in the track"""
        return self._keyboxes[0] if len(self._keyboxes) > 0 else None

    def endbox(self):
        """The endbox is the last box in the track"""
        return self._keyboxes[-1] if len(self._keyboxes) > 0 else None

    def loop_closure_distance(self):
        """The loop closure track distance is the Euclidean distance in pixels between the start frame bounding box and end frame bounding box"""
        return self.startbox().dist(self.endbox()) if not self.isdegenerate() else None

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

    def iou(self, other, dt=1):
        """Compute the spatial IoU between two tracks as the mean IoU per frame in the range (self.startframe(), self.endframe())"""
        return self.rankiou(other, rank=len(self), dt=dt)

    def segment_maxiou(self, other, startframe, endframe):
        """Return the maximum framewise bounding box IOU between self and other in the range (startframe, endframe)"""
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"
        assert startframe < endframe
        return max([self[k].iou(other[k]) if (self[k] is not None) else 0 for k in range(startframe, endframe)])
    
    def maxiou(self, other, dt=1):
        """Compute the maximum spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe())"""        
        return self.rankiou(other, rank=1, dt=dt)

    def fragmentiou(self, other, dt=5):
        """A fragment is a track that is fully contained within self"""
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"        
        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())
        return float(np.min([self[min(k,endframe)].iou(other[min(k,endframe)]) for k in range(startframe, endframe, dt)])) if (other.startframe() >= self.startframe() and other.endframe() <= self.endframe() and endframe > startframe) else 0
        
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
        
    def rankiou(self, other, rank, dt=1):
        """Compute the mean spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe()) using only the top-k (rank) frame overlaps
           Sample tracks at endpoints and n uniformly spaced frames or a stride of dt frames.  
        
           - rank [>1]:  The top-k best IOU overlaps to average when computing the rank IOU
           - This is useful for track continuation where the box deforms in the overlapping segment at the end due to occlusion. 
           - This is useful for track correspondence where a ground truth box does not match an estimated box precisely (e.g. loose box, non-visually grounded box)
           - This is the robust version of segmentiou.
           - Use percentileiou to determine the rank based a fraction of the length of the overlap, which will be more efficient for long tracks
        """
        assert rank >= 1 and rank <= len(self)
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        assert dt >= 1
        frames = [self.startframe()] + list(range(self.startframe()+dt, self.endframe(), dt)) + [self.endframe()]
        return float(np.mean(sorted([self[k].iou(other[k]) if (self.during(k) and other.during(k)) else 0.0 for k in frames])[-rank:]))

    def percentileiou(self, other, percentile, samples=100):
        """Percentile iou returns rankiou for rank=percentile*len(overlap(self, other))
        
           -other [Track]
           -percentile [0,1]:  The top-k best overlaps to average when computing rankiou
           -samples:  The number of uniformly spaced samples to take along the track for computing the rankiou
        """
        assert percentile > 0 and percentile <= 1
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"

        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())
        segmentlen = endframe - startframe
        dt = max(1, int(np.floor(segmentlen/samples)))
        return self.rankiou(other, max(1, int(segmentlen*percentile)), dt=dt) if segmentlen > 0 else 0

    def segment_percentileiou(self, other, percentile, samples=100):
        """percentiliou on the overlapping segment with other"""
        assert percentile > 0 and percentile <= 1
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"

        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())
        segmentlen = endframe - startframe
        rank = int(segmentlen*percentile)
        dt = max(1, int(np.floor(segmentlen/samples)))
        iou = sorted([self[min(k,endframe)].iou(other[min(k,endframe)]) for k in range(startframe, endframe, dt)]) if endframe > startframe else []
        return float(np.mean(iou[-rank:]) if endframe > startframe else 0.0)


    def segment_percentilecover(self, other, percentile, samples=100):
        """percentile cover on the overlapping segment with other"""
        assert percentile > 0 and percentile <= 1
        assert isinstance(other, Track), "invalid input - Must be vipy.object.Track()"

        startframe = max(self.startframe(), other.startframe())
        endframe = min(self.endframe(), other.endframe())
        segmentlen = endframe - startframe
        rank = int(segmentlen*percentile)
        dt = max(1, int(np.floor(segmentlen/samples)))
        bblist = [(self[min(k,endframe)], other[min(k,endframe)]) for k in range(startframe, endframe, dt)] if endframe > startframe else []
        cover = [max(bbself.cover(bbother), bbother.cover(bbself)) for (bbself, bbother) in bblist]
        return float(np.mean(cover[-rank:]) if endframe > startframe else 0.0)

    def union(self, other, overlap='average'):
        """Compute the union of two tracks.  Overlapping boxes between self and other:
        
           Inputs
             - average [bool]:  average framewise interpolated boxes at overlapping keyframes
             - replace [bool]:  replace the box with other if other and self overlap at a keyframe
             - keep [bool]:  keep the box from self (discard other) at a keyframe

        """
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        assert other.category() == self.category(), "Category mismatch"
        assert overlap in ['average', 'replace', 'keep'], "Invalid input - 'overlap' must be in [average, replace, keep]"
        T = self.clone()
        keyframes = sorted(set(T._keyframes+other._keyframes))
        T._keyboxes = [((self[k].average(other[k]) if (overlap == 'average') else (self[k] if (overlap == 'keep') else other[k]))
                        if (self.during(k) and other.during(k)) else 
                        (self[k] if (self.during(k) and not other.during(k)) else (other[k])))
                       for k in keyframes] 
        T._keyframes = keyframes
        return T  


    def average(self, other):
        """Compute the average of two tracks by the framewise interpolated boxes at the keyframes of this track"""
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        assert other.category() == self.category(), "Category mismatch"
        T = self.clone()
        T._keyboxes = [(self[k].average(other[k]) 
                        if (self.during(k) and other.during(k)) else (self[k] if (self.during(k) and not other.during(k)) else (other[k])))
                       for k in T._keyframes]  
        return T  

    def temporal_distance(self, other):
        """The temporal distance between two tracks is the minimum number of frames separating them"""
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        return max(max(self.startframe() - other.endframe(), other.startframe() - self.endframe()), 0)

    def smooth(self, width):
        """Track smoothing by averaging neighboring keyboxes"""
        assert isinstance(width, int) and width > 0
        if len(self._keyboxes) > width:
            self._keyboxes = [bb.clone().average(bbnbrs) for (bb, bbnbrs) in zip(self._keyboxes, chunklistwithoverlap(self._keyboxes, width, width-1))] 
        return self

    def smoothshape(self, width):
        """Track smoothing by averaging width and height of neighboring keyboxes"""
        assert isinstance(width, int) and width > 0
        if len(self._keyboxes) > width:
            self._keyboxes = [bb.clone().averageshape(bbnbrs) for (bb, bbnbrs) in zip(self._keyboxes, chunklistwithoverlap(self._keyboxes, width, width-1))]
        return self

    def medianshape(self, width):
        """Track smoothing by median width and height of neighboring keyboxes"""
        assert isinstance(width, int) and width > 0
        if len(self._keyboxes) > width:
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

    def linear_extrapolation(self, k, shape=False, dt=30):
        """Track extrapolation by linear fit.
        
           * Requires at least 2 keyboxes.
           * Returned boxes may be degenerate.
           * shape=True then both the position and shape (width, height) of the box is extrapolated
        """
        if self.during(k):
            return self[k]
        elif len(self._keyboxes) == 1:
            return self.nearest_keybox(k)
        else:
            n = self.endframe() if k > self.endframe() else self.startframe()+1
            d = self.endbox().clone() if k > self.endframe() else self.startbox().clone()
            (vx, vy) = self.shape_invariant_velocity(n, dt=dt) if not shape else self.velocity(n, dt=dt)
            (vw, vh) = (self.velocity_w(n, dt=dt), self.velocity_h(n, dt=dt)) if shape else (0,0)
            d = d.translate((k-n)*vx, (k-n)*vy)
            return d if not shape else d.top( ((k-n)*vh)/2.0).bottom( ((k-n)*vh)/2.0).left( ((k-n)*vw)/2.0).right( ((k-n)*vw)/2.0)
            
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
        """Round the coordinates of all boxes so that they have n significant digits for efficient serialization"""
        self._keyboxes = [bb.significant_digits(n) for bb in self._keyboxes]
        return self

    def bearing(self, f, dt=30, minspeed=1):
        """The bearing of a track at frame f is the angle of the velocity vector relative to the (x,y) image coordinate frame, in radians [-pi, pi]"""
        v = self.shape_invariant_velocity(f, dt)
        return float(np.arctan2(v[1], v[0])) if self.speed(f, dt) > minspeed else None  # atan2(y,x)

    def bearing_change(self, f1=None, f2=None, dt=30, minspeed=1):
        """The bearing change of a track from frame f1 (or start) and frame f2 (or end) is the relative angle of the velocity vectors in radians [-pi,pi]"""
        dt = min(dt, len(self))
        B = [self.bearing(k, dt=dt, minspeed=minspeed) for k in range(f1 if f1 is not None else self.startframe(), f2 if f2 is not None else self.endframe())]
        B = [b for b in B if b is not None]  # valid bearing estimates only
        dr = np.sum(np.diff(B)) if len(B) > 0 else 0  # cumulative bearing angle change 
        return float(dr if np.abs(dr)<=np.pi else ((2*np.pi - dr) if (dr > np.pi) else (2*np.pi + dr)))

    def acceleration(self, f, dt=30):
        """Return the (x,y) track acceleration magnitude at frame f computed using central finite differences of velocity"""
        (u, v) = (self.shape_invariant_velocity(f-dt, dt), self.shape_invariant_velocity(f+2*dt, dt))  # ((f-2*dt, (f-dt)), (f+dt, f+2*dt))
        (ax, ay) = ((v[0] - u[0])/float(2*dt), (v[1] - u[1])/float(2*dt))
        return float(np.sqrt(ax**2 + ay**2))  # acceleration magnitude in pixels
        
    def velocity(self, f, dt=30):
        """Return the (x,y) track velocity at frame f in units of pixels per frame computed by mean finite difference of the box centroid"""
        return (self.velocity_x(f, dt), self.velocity_y(f, dt))

    def speed(self, f, dt=30):
        (u,v) = self.shape_invariant_velocity(f, dt)
        return float(np.sqrt(u**2 + v**2))
    
    def boxmap(self, f):
        """Apply the lambda function to each keybox"""
        self._keyboxes = [f(bb) for bb in self._keyboxes]
        return self

    def shape_invariant_velocity(self, f, dt=30):
        """Return the (x,y) track velocity at frame f in units of pixels per frame computed by minimum mean finite differences of any box corner independent of changes in shape, over a finite time window of [f-dt, f]"""
        assert f >= 0 and dt > 0
        if len(self) < 2 or not (self.during(f) and self.during(f-dt)) :
            return (0,0)
        
        kb = [((f-dt), self.linear_interpolation(f-dt, id=False))] + [(kf, bb) for (kf,bb) in zip(self._keyframes, self._keyboxes) if (kf > f-dt) and (kf < f)]
        (kfe, bbe) = (f, self.linear_interpolation(f, id=False))
        vx = float((1.0/len(kb))*sum([min([(bbe._xmin - bb._xmin), (bbe._xmax - bb._xmax)], key=abs)/float(kfe-kf) for (kf,bb) in kb]))
        vy = float((1.0/len(kb))*sum([min([(bbe._ymin - bb._ymin), (bbe._ymax - bb._ymax)], key=abs)/float(kfe-kf) for (kf,bb) in kb]))
        return (vx, vy)

    def velocity_x(self, f, dt=30):
        """Return the left/right velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid"""
        assert f >= 0 and dt > 0
        return float(np.mean([(self[f].centroid_x() - self[f-k].centroid_x())/float(k) for k in range(1,dt) if self.during(f-k)])) if (self.during(f-1) and self.during(f)) else 0

    def velocity_y(self, f, dt=30):
        """Return the up/down velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid"""
        assert f >= 0 and dt > 0
        return float(np.mean([(self[f].centroid_y() - self[f-k].centroid_y())/float(k) for k in range(1,dt) if self.during(f-k)])) if (self.during(f-1) and self.during(f)) else 0

    def velocity_w(self, f, dt=30):
        """Return the width velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return float(np.mean([(self[f].width() - self[f-k].width())/float(k) for k in range(1,dt) if self.during(f-k)])) if self.during(f-1) else 0

    def velocity_h(self, f, dt=30):
        """Return the height velocity at frame f in units of pixels per frame computed by finite difference"""
        assert f >= 0 and dt > 0 and self.during(f)
        return float(np.mean([(self[f].height() - self[f-k].height())/float(k) for k in range(1,dt) if self.during(f-k)])) if self.during(f-1) else 0
    
    def nearest_keyframe(self, f):
        """Nearest keyframe to frame f"""
        assert len(self._keyframes) > 0
        return self._keyframes[int(np.abs(np.array(self._keyframes) - f).argmin())]

    def nearest_keybox(self, f):
        """Nearest keybox to frame f"""
        assert len(self._keyframes) > 0
        return self._keyboxes[int(np.abs(np.array(self._keyframes) - f).argmin())]  # by-reference
    
    def ismoving(self, startframe=None, endframe=None, mincover=0.9):
        """Is the track moving in the frame range (startframe,endframe)?"""
        (bbs, bbe) = (self[max(self.startframe(), startframe)] if startframe is not None else self.startbox(), self[min(self.endframe(), endframe)] if endframe is not None else self.endbox())
        return (bbs.maxcover(bbe) < mincover) if (bbs is not None and bbe is not None) else False

    
def non_maximum_suppression(detlist, conf, iou, bycategory=False, cover=None, gridsize=(6,9)):
    """Compute greedy non-maximum suppression of a list of vipy.object.Detection() based on spatial IOU threshold (iou) and cover threhsold (cover) sorted by confidence (conf)"""
    assert all([isinstance(d, Detection) for d in detlist])
    assert all([d.confidence() is not None for d in detlist])
    assert conf>=0 and iou>=0 and iou<=1
    assert cover is None or (cover>=0 and cover<=1)
    assert isinstance(gridsize, tuple) and len(gridsize) == 2
        
    suppressed = set([])
    detlist = [d for d in detlist if d.confidence() > conf and not d.isdegenerate()]  # valid
    detlist.sort(key=lambda d: d.confidence(), reverse=True)  # biggest to smallest, in-place
    grid = detlist[0].clone().union(detlist).grid(gridsize[0], gridsize[1]) if len(detlist) > 0 else []
    bbidx = [set([k for (k,bbg) in enumerate(grid) if (((bbg._xmax if bbg._xmax < bb._xmax else bb._xmax) - (bbg._xmin if bbg._xmin > bb._xmin else bb._xmin)) > 0 and
                                                       ((bbg._ymax if bbg._ymax < bb._ymax else bb._ymax) - (bbg._ymin if bbg._ymin > bb._ymin else bb._ymin)) > 0)])
             for bb in detlist]  # spatial index, without the function call overhead of bbg.hasintersection(bb)
    #bbidx = [set([k for (k,bbg) in enumerate(grid) if bbg.hasintersection(bb)]) for bb in detlist]  # spatial index, equivalent to above but slower
    
    area = [bb.area() for bb in detlist]
    for (i, di) in enumerate(detlist):
        if i in suppressed:
            continue
        for (j, dj) in enumerate(islice(detlist, i+1, None), start=i+1):  # no-copy, equivalent to detlist[i+1:]
            if ((j not in suppressed) and
                (bycategory is False or di._label == dj._label) and
                (not bbidx[i].isdisjoint(bbidx[j])) and
                ((cover is not None and di.hasintersection(dj, maxcover=cover, area=area[i], otherarea=area[j])) or di.hasintersection(dj, iou=iou, area=area[i], otherarea=area[j]))):  
                suppressed.add(j)
    detlist_nms = [d for (j,d) in enumerate(detlist) if j not in suppressed]  # filter
    detlist_nms.sort(key=lambda x: x.confidence())  # smallest to biggest confidence for display layering, in-place
    return detlist_nms


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

def RandomDetection(W=640, H=480):
    return Detection(xmin=np.random.rand()*W, ymin=np.random.rand()*H, width=np.random.rand()*100, height=np.random.rand()*100, category=str(np.random.rand()), confidence=np.random.rand())


