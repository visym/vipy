import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist
import uuid
import copy


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
            strlist.append('conf=%1.3f')
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
            self._label = category
            return self

    def shortlabel(self, label=None):
        """A optional shorter label string to show in the visualizations, defaults to category()"""        
        if label is not None:
            self._shortlabel = label
            return self
        else:
            return self._shortlabel

    def label(self, label):
        """Alias for category"""
        return self.category(label)

    def id(self):
        return self._id

    def clone(self):
        return copy.deepcopy(self)
    

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

    def __init__(self, keyframes, boxes, category=None, label=None, confidence=None, framerate=None, interpolation='linear', boundary='strict', shortlabel=None, attributes=None):

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
                
        self._id = uuid.uuid1().hex
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
            self._keyframes = list(keyframes)
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
        """The length of a track is the total number of interpolated frames"""
        return self.endframe() - self.startframe() + 1

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
        
    def keyframes(self):
        """Return keyframe frame indexes where there are track observations"""
        return self._keyframes

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
        return np.min(self._keyframes)

    def endframe(self):
        return np.max(self._keyframes)

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
        """A optional shorter label string to show in the visualizations"""                
        if label is not None:
            self._shortlabel = label
            return self
        else:
            return self._shortlabel

    def during(self, k):
        """Is frame during the time interval (startframe, endframe) inclusive?"""        
        return k >= self.startframe() and k <= self.endframe()

    def offset(self, dt=0, dx=0, dy=0):
        self._keyboxes = [bb.offset(dx, dy) for bb in self._keyboxes]
        self._keyframes = list(np.array(self._keyframes) + dt)
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

    def id(self):
        return self._id

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
        self._keyframes = [f for (f,bb) in zip(self._keyframes, self._keyboxes) if f>=startframe and f<=endframe]  # may be empty
        self._keyboxes = [bb for (f,bb) in zip(self._keyframes, self._keyboxes) if f>=startframe and f<=endframe]  # may be empty
        if len(self._keyframes) == 0 or len(self._keyboxes) == 0:
            raise ValueError('Track does not contain any keyboxes within the requested frames (%d,%d)' % (startframe, endframe))
        self._boundary = 'strict'
        return self

    def iou(self, other):
        """Compute the spatial IoU between two tracks as the mean IoU per frame in the range (self.startframe(), self.endframe())"""
        assert isinstance(other, Track), "Invalid input - must be vipy.object.Track()"
        return np.mean([self[k].iou(other[k]) if self.during(k) and other.during(k) else 0.0 for k in range(self.startframe(), self.endframe())])

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


class Activity(object):
    """vipy.object.Activity class
    
    An activity is a grouping of one or more tracks involved in an activity within a given startframe and endframe.
    The activity occurs at a given (startframe, endframe), where these frame indexes are extracted at the provided framerate.
    All objects are passed by reference with a globally unique track ID, for the tracks involved with the activity.  This 
    is done since tracks can exist after an activity completes, and that tracks should update the spatial transformation of boxes.
    The shortlabel defines the string shown on the visualization video.

    Valid constructors

    >>> t = vipy.object.Track(category='Person').add(...))
    >>> a = vipy.object.Activity(startframe=0, endframe=10, category='Walking', tracks={t.id():t})

    """
    def __init__(self, startframe, endframe, framerate=None, label=None, shortlabel=None, category=None, tracks=None, attributes=None):
        assert not (label is not None and category is not None), "Activity() Constructor requires either label or category kwargs, not both"
        assert startframe < endframe, "Start frame must be strictly less than end frame"
        assert tracks is None or isinstance(tracks, dict), "Tracks must be a dictionary {trackid:vipy.object.Track()}"
        assert tracks is None or all([isstring(k) for (k,v) in tracks.items()]) and all([isinstance(v, Track) for (k,v) in tracks.items()]), "Invalid tracks - Must be a dictionary of {str(trackid):vipy.object.Track()}"        
        assert tracks is None or all([any([t.during(f) for f in range(startframe, endframe)]) for t in tracks.values()]), "All tracks must be be present in at least one frame when this activity occurs"
    
        self._id = uuid.uuid1().hex
        self._startframe = startframe
        self._endframe = endframe
        self._framerate = framerate
        self._label = category if category is not None else label        
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._tracks = tracks if tracks is not None else {}

        self.attributes = attributes if attributes is not None else {}            
        
    def __len__(self):
        """Return activity length in frames"""
        return self.endframe() - self.startframe()

    def __getitem__(self, k):
        """Return a list Detection() objects interpolated at frame k if during activity, otherwise return None"""
        return [t[k] for (i,t) in self._tracks.items() if t[k] is not None] if self.during(k) else None
    
    def __repr__(self):
        return str('<vipy.activity: category="%s", frames=(%d,%d), tracks=%s>' % (self.category(), self.startframe(), self.endframe(), len(self.tracks())))

    def dict(self):
        return {'id':self._id, 'label':self.category(), 'shortlabel':self.shortlabel(), 'startframe':self._startframe, 'endframe':self._endframe, 'attributes':self.attributes, 'framerate':self._framerate,
                'tracks':[t.dict() for (k,t) in self._tracks.items()]}
    
    def startframe(self):
        return self._startframe
    
    def endframe(self):
        return self._endframe

    def middleframe(self):
        return int(np.round((self.endframe() - self.startframe()) / 2.0)) + self.startframe()

    def framerate(self, fps):
        """Resample (startframe, endframe) from known original framerate set by constructor to be new framerate fps"""        
        assert self._framerate is not None, "Framerate conversion requires that the framerate is known for current activities.  This must be provided to the vipy.object.Activity() constructor."
        (self._startframe, self._endframe) = [int(np.round(f*(fps/float(self._framerate)))) for f in (self._startframe, self._endframe)]
        self._framerate = fps
        return self
    
    def category(self, label=None):
        if label is not None:
            self._label = label
            return self
        else:
            return self._label

    def label(self, label):
        """Alias for category"""
        return self.category(label)

    def categories(self):
        """Return a set of categories for the activities and objects in this activity"""
        return set([self.category()] + [t.category() for t in self.tracks().values()])
        
    def shortlabel(self, label=None):
        """A optional shorter label string to show in the visualizations"""                
        if label is not None:
            self._shortlabel = label
            return self
        else:
            return self._shortlabel

    def add(self, track):
        """Add the track by reference to the track list for this activity, so that if the track is changed externally it is reflected here"""
        assert isinstance(track, Track), "Invalid input - must be vipy.object.Track"
        assert track.during(self.startframe()) or track.during(self.endframe()) or track.during(self.middleframe()), "The track must be present during the activity"
        self._tracks[track.id()] = track
        return self
        
    def tracks(self, tracks=None):
        """Returns a track dictionary, tracks are referenced in the dictionary and are mutable"""
        if tracks is not None:
            assert isinstance(tracks, dict), "Tracks must be dictionary of {str(trackid):vipy.object.Track()}"
            self._tracks = tracks
            return self
        else:
            return self._tracks

    def hastrack(self, track):
        """Is the track part of the activity?"""
        assert isstring(track) or isinstance(track, Track), "Invalid input - Must be a vipy.object.Track().id() or vipy.object.Track()"
        trackid = track.id() if isinstance(track, Track) else track
        return any([tid == trackid for tid in self._tracks.keys()])
            
    def during(self, frame):
        """Is frame during the time interval (startframe, endframe) inclusive?"""
        return int(frame) >= self._startframe and int(frame) <= self._endframe

    def spatial_iou(self, other):
        """Return the mean spatial intersection over union of two activities as the mean spatial IoU for the union of tracks at each frame during (startframe, endframe)
           Note that we cannot do the IoU of individual tracks because there is no way to correspond tracks within an activity, since an activity may have more than one 
           track with the same category.  
        """
        assert isinstance(other, Activity), "Invalid input - must be vipy.object.Activity()"
        return np.mean([bbi.union(self[k]).iou(bbj.union(other[k])) if (bbj is not None and bbi is not None) else 0.0 for k in range(self.startframe(), self.endframe()) for bbi in self[k] for bbj in other[k]])

    def temporal_iou(self, other):
        """Return the temporal intersection over union of two activities"""
        assert isinstance(other, Activity), "Invalid input - must be vipy.object.Activity()"
        t_start = min(self.startframe(), other.startframe())
        t_end = max(self.endframe(), other.endframe())
        t_union = float(t_end - t_start)
        
        t_start = max(self.startframe(), other.startframe())
        t_end = min(self.endframe(), other.endframe())
        t_intersection = float(t_end - t_start)
        
        return (t_intersection / t_union) if t_intersection > 0 else 0
    
    def offset(self, dt):
        self._startframe = self._startframe + dt
        self._endframe = self._endframe + dt
        self._tracks = {ti:t.offset(dt=dt) for (ti,t) in self._tracks.items()}
        return self
    
    def id(self):
        return self._id

    def boundingbox(self):
        """The bounding box of an activity is the smallest bounding box for all tracks in the activity (inclusive of start and endframes), or None of there are no boxes""" 
        boxes = [t.clone().clip(self.startframe(), self.endframe()+1).boundingbox() for (i,t) in self.tracks().items()]
        return boxes[0].clone().union(boxes[1:]) if len(boxes)>0 else None

    def imagebox(self, width, height):
        """The image box of an activity is the smallest bounding box for all tracks in the activity (inclusive of start and endframes) that is within the image rectangle (width, height), or None if there are no boxes""" 
        bb = self.boundingbox()
        return None if bb is None or not bb.hasoverlap(width=width, height=height) else bb.imclipshape(width, height)

    def clone(self):
        return copy.deepcopy(self)
    
