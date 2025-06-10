import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist, shortuuid
import uuid
import copy
from vipy.object import Track

try:
    import ujson as json  # faster
except ImportError:
    import json


class Activity(object):
    """vipy.object.Activity class
    
    An activity is a grouping of one or more tracks involved in an activity within a given startframe and endframe.
    The activity occurs at a given (startframe, endframe), where these frame indexes are extracted at the provided framerate.
    All objects are passed by reference with a globally unique track ID, for the tracks involved with the activity.  This 
    is done since tracks can exist after an activity completes, and that tracks should update the spatial transformation of boxes.
        
    Valid constructors

    ```python 
    t = vipy.object.Track(category='Person')
    a = vipy.object.Activity(startframe=0, endframe=10, category='Walking', tracks=t.id())
    a = vipy.object.Activity(startframe=0, endframe=10, category='Walking', tracks=t)    
    ```

    Note.. shortlabel is kepy for backwards compatibility and will be deprecated
    
    """
    __slots__ = ['_id', '_startframe', '_endframe', '_framerate', '_label', '_trackid', '_actorid', 'attributes']
    def __init__(self, startframe, endframe, framerate=30, label=None, category=None, tracks=None, attributes=None, actorid=None, confidence=None, id=None, shortlabel=None):
        assert not (label is not None and category is not None), "Activity() Constructor requires either label or category kwargs, not both"
        assert startframe <= endframe, "Start frame must be less than or equal to end frame"
        assert framerate is not None, "initial framerate is required for framerate conversion"
        if tracks:
            tracks = [v for (k,v) in tracks.items()] if isinstance(tracks, dict) else tracks  # backwards compatible, track dictionary input
            assert (all([isstring(t) for t in tracks]) or all([isinstance(t, Track) for t in tracks])), "Invalid track input"
            if all([isinstance(t, Track) for t in tracks]):
                assert any([any([t.during(f) for f in range(startframe, endframe)]) for t in tracks]), "At least one track must be be present in at least one frame when this activity occurs"
                tracks = [t.id() for t in tracks]  
        trackid = set(tracks) if tracks is not None else (set([actorid]) if actorid is not None else set([]))  # only store IDs, not tracks
        
        if tracks is not None and actorid is not None and actorid not in trackid:
            trackid.add(actorid)

        self._id = shortuuid() if id is None else id  # use provided
        self._startframe = int(startframe)
        self._endframe = int(endframe)
        self._framerate = float(framerate)
        self._label = category if category is not None else label        
        self._trackid = trackid
        self._actorid = actorid

        self.attributes = attributes.copy() if attributes is not None else {}  # shallow copy
        if confidence is not None:
            self.attributes['confidence'] = float(confidence)

    def hasattribute(self, k):
        return k in self.attributes

    def confidence(self, c=None):
        if c is not None:
            self.attributes['confidence'] = float(c)
            return self
        else:
            return self.attributes['confidence'] if 'confidence' in self.attributes else None
    
    @classmethod
    def from_json(obj, s):
        d = json.loads(s) if not isinstance(s, dict) else s
        d = {k.lstrip('_'):v for (k,v) in d.items()}  # prettyjson (remove "_" prefix to attributes)                                        
        return obj(startframe=int(d['startframe']),
                   endframe=int(d['endframe']),
                   framerate=d['framerate'] if 'framerate' in d else None,
                   category=d['label'] if 'label' in d else None,
                   tracks=d['trackid'] if 'trackid' in d else None,
                   attributes=d['attributes'] if 'attributes' in d else None,
                   actorid=d['actorid'] if 'actorid' in d else None,
                   id=d['id'] if 'id' in d else None)
                
    def __len__(self):
        """Return activity length in frames, or zero if degenerate"""
        return max(0, self.endframe() - self.startframe())

    def duration(self, s=None, centered=False):
        """The length of the activity in seconds.

        Args:
            s: [float] The number of seconds for this activity, starting at the startframe
            centered: [bool] If true, then set the duration centered on the middle frame

        Returns:
            The duration in seconds of this activity object (if s=None)
            This activity object with the requested duration (if s!=None)
        """
        assert self.framerate() is not None, "Framerate must be set in constructor"
        if s is None:
            return len(self) / float(self.framerate())
        else:
            return (self.endframe(self.startframe() + int(round(s*self.framerate()))) if not centered else 
                    self.truncate(startframe=self.middleframe()-int(np.ceil(s*self.framerate())//2), endframe=self.middleframe()+int(np.ceil(s*self.framerate())//2)))
        
    def __repr__(self):
        return str('<vipy.activity.Activity: category="%s", frames=(%d,%d), tracks=%s%s>' % (self.category(), self.startframe(), self.endframe(), len(self.trackids()), '' if self.confidence() is None else ', confidence=%1.2f' % self.confidence()))

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(s=None, encode=False)

    def __json__(self):
        """Serialization method for json package"""
        return self.json(encode=True)
    
    def json(self, encode=True):
        d = {k.lstrip('_'):getattr(self, k) for k in Activity.__slots__ if getattr(self, k) is not None}  # prettyjson (remove "_" prefix to attributes)          
        d = {k:v if k != 'trackid' else tuple(v) for (k,v) in d.items()}  # sets are non-serializable
        return json.dumps(d) if encode else d
    
    def actorid(self, actorid=None):
        if actorid is None:
            return self._actorid
        else:
            self._actorid = actorid
            if actorid not in self._trackid:
                self._trackid.add(actorid) if isinstance(self._trackid, set) else self._trackid.append(actorid)                    
            return self

    def startframe(self, f=None):
        if f is None:
            return self._startframe
        else:
            self._startframe = f
            return self
    
    def endframe(self, f=None):
        if f is None:
            return self._endframe
        else:
            self._endframe = int(f)
            return self

    def middleframe(self):
        """Return the middle frame number of the activity"""
        return int(np.round((self._endframe - self._startframe) / 2.0)) + self._startframe

    def _set_framerate(self, fps):
        """Override framerate conversion and just set the framerate attribute.  This should really be set only in constructor.  Use with caution!"""
        self._framerate = float(fps)
        return self

    def framerate(self, fps=None, speed=None):
        """Resample (startframe, endframe) from known original framerate set by constructor to be new framerate fps"""        
        if fps is None and speed is None:
            return self._framerate
        else:
            assert fps is not None or speed is not None, "Invalid input"
            assert not (fps is not None and speed is not None), "Invalid input"
            assert speed is None or speed > 0, "Invalid speed, must specify speed multiplier s=1, s=2 for 2x faster, s=0.5 for half slower"                    
            assert fps is None or fps > 0, "Invalid framerate"
            assert self._framerate is not None, "Framerate conversion requires that the framerate is known for current activities.  This must be provided to the vipy.object.Activity() constructor."

            fps = float(fps) if fps is not None else (1.0/speed)*self._framerate            
            (self._startframe, self._endframe) = [int(np.round(f*(fps/float(self._framerate)))) for f in (self._startframe, self._endframe)]
            self._framerate = float(fps)
            return self
    
    def category(self, label=None):
        """Change the label to the new label """
        if label is not None:
            self._label = label
            return self
        else:
            return self._label

    def categoryif(self, ifcategory, tocategory=None):
        """If the current category is equal to ifcategory, then change it to newcategory.

        Args:
            
            ifcategory [dict, str]: May be a dictionary {ifcategory:tocategory}, or just an ifcategory
            tocategory [str]:  the target category 

        Returns:
        
            this object with the category changed.

        .. note:: This is useful for converting synonyms such as self.categoryif('person_sits', 'person_sitting')
        """
        assert (isinstance(ifcategory, dict) and tocategory is None) or tocategory is not None

        if isinstance(ifcategory, dict):
            for (k,v) in ifcategory.items():
                self.categoryif(k, v)
        elif self.category() == ifcategory:
            self.category(tocategory)
        return self

    def label(self, label=None):
        """Alias for category"""
        return self.category(label)

    def add(self, track):
        """Add the track id for the track to this activity, so that if the track is changed externally it is reflected here"""
        assert isinstance(track, Track), "Invalid input - must be vipy.object.Track"
        assert self.during_interval(track.startframe(), track.endframe()), "The track must be present during the activity"
        if track.id() not in self._trackid:
            self._trackid = list(self._trackid)
            self._trackid.append(track.id())
        return self

    def addid(self, trackid):
        """Add the track id for the track to this activity, so that if the track is changed externally it is reflected here"""
        if trackid not in self._trackid:
            self._trackid = list(self._trackid)
            self._trackid.append(trackid)
        return self
    
    def tracks(self):
        """alias for trackids"""
        return self.trackids()

    def cleartracks(self):
        """Remove all track IDs from this activity"""
        self._trackid = []
        return self

    def trackids(self):
        """Return a set of track IDs associated with this activity"""
        return set(self._trackid)

    def hasoverlap(self, other, threshold=0):
        """Return true if the temporal_iou is greater than the provided threshold between self and other Track or other Activity"""
        assert isinstance(other, Activity) or isinstance(other, Track), "Invalid input"
        assert threshold >= 0 and threshold <= 1, "Invalid temporal IOU threshold"
        return (((min(self._endframe, other.endframe()) - max(self._startframe, other.startframe())) > 0) if threshold == 0 else
                self.temporal_iou(other) > threshold)
        
    def isneighbor(self, other, framegate=10):
        return self.temporal_iou(other.clone().temporalpad(framegate)) > 0 

    def hastrack(self, track):
        """Is the track part of the activity?"""
        assert isstring(track) or isinstance(track, Track), "Invalid input - Must be a vipy.object.Track().id() or vipy.object.Track()"
        trackid = track.id() if isinstance(track, Track) else track
        return trackid in self._trackid

    def hastrackoverlap(self, track):
        """is the activity occurring during the interval when the track is occurring and is this track assigned to the activity?"""
        assert isinstance(track, Track)
        return self.hastrack(track) and self.temporal_iou(track) > 0

    def append(self, newtrack):
        """Append newtrack to this activity and set as actorid()"""
        assert isinstance(newtrack, Track), "Invalid input - must be vipy.object.Track"
        self._trackid.add(newtrack.id()) if isinstance(self._trackid, set) else self._trackid.append(newtrack.id())
        self.actorid(newtrack.id())
        return self

    def trackfilter(self, f):
        """Remove all tracks such that the lambda function f(trackid) resolves to False"""
        assert callable(f)
        self._trackid = [tid for tid in self._trackid if f(tid)]
        if self.actorid() not in self._trackid:
            self._actorid = None
        return self

    def replace(self, oldtrack, newtrack):
        """Replace oldtrack with newtrack if present in self._tracks.  Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change"""
        assert isinstance(oldtrack, Track) and isinstance(newtrack, Track), "Invalid input - must be vipy.object.Track"
        if self.hastrack(oldtrack):
            self._trackid.discard(oldtrack.id())
            self._trackid.add(newtrack.id())
            if self.actorid() == oldtrack.id():
                self.actorid(newtrack.id())
        return self    

    def replaceid(self, oldtrackid, newtrackid):
        """Replace oldtrack with newtrack if present in self._tracks.  Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change"""
        if self.hastrack(oldtrackid):
            ti = set(self._trackid)
            ti.discard(oldtrackid)
            ti.add(newtrackid)
            if self.actorid() == oldtrackid:
                self.actorid(newtrackid)
            self._trackid = list(ti)
        return self    
    
    def during(self, frame):
        """Is frame during the time interval (startframe, endframe) inclusive?"""
        return int(frame) >= self._startframe and int(frame) <= self._endframe

    def during_interval(self, startframe, endframe, inclusive=False):
        """Is the activity occurring for any frames within the interval [startframe, endframe) (non-inclusive of endframe)?"""
        ef = endframe+(0 if not inclusive else 1)
        return (startframe >= self._startframe and startframe <= self._endframe) or (ef >= self._startframe and ef <= self._endframe) or (startframe <= self._startframe and ef >= self._endframe)

    
    def union(self, other, confweight=0.5, maxconf=False):
        """Compute the union of the new activity other to this activity by updating the start and end times and computing the mean confidence.
        
           -Note: other must have the same category and track IDs as self
           -confweight [0,1]:  the convex combinatiopn weight applied to the new activity 
        """
        assert isinstance(other, Activity), "Invalid input"
        assert self._actorid == other._actorid, "Actor ID must be the same"
        assert self._label == other._label, "Assigned activity is a different category"
        assert self._framerate == other._framerate, "Invalid input"
        assert confweight >= 0 and confweight <= 1, "Confidence weight must be [0,1]"

        self.startframe(min(other._startframe, self._startframe))
        self.endframe(max(other._endframe, self._endframe))
        if other.confidence() is not None and self.confidence() is not None:
            self.confidence(float((1.0-confweight)*self.confidence() + confweight*other.confidence()) if not maxconf else float(max(self.confidence(), other.confidence())))  # running mean confidence or max
        return self
        
    def temporal_iou(self, other):
        """Return the temporal intersection over union of two activities or this activity and a track"""
        assert isinstance(other, Activity) or isinstance(other, Track), "Invalid input - must be vipy.activity.Activity or vipy.object.Track"
        assert self._framerate == other._framerate, "invalid input - framerate must match"
        (sf, ef) = (other._startframe, other._endframe) if isinstance(other, Activity) else (other.startframe(), other.endframe())  # attribute access is faster than methods
        
        t_start = min(self._startframe, sf)
        t_end = max(self._endframe, ef)
        t_union = float(t_end - t_start)
        
        t_start = max(self._startframe, sf)
        t_end = min(self._endframe, ef)
        t_intersection = float(t_end - t_start)
        
        return (t_intersection / t_union) if t_intersection > 0 else 0
    
    def offset(self, dt):
        dt = dt if isinstance(dt, int) else int(np.round(dt*self.framerate()))
        self._startframe = int(self._startframe + dt)
        self._endframe = int(self._endframe + dt)
        return self

    def truncate(self, start=None, end=None):
        """Truncate the activity so that it is between start and end"""
        startframe = int(np.round(start*self.framerate())) if isinstance(start, float) else start
        endframe = int(np.round(end*self.framerate())) if isinstance(end, float) else end        
        
        self._startframe = self._startframe if startframe is None else max(self._startframe, startframe)
        self._endframe = self._endframe if endframe is None else min(self._endframe, endframe)
        self._endframe = self._endframe if self._endframe > self._startframe else self._startframe  # degenerate truncation
        return self
            
    def id(self, newid=None):
        if newid is None:
            return self._id
        else:
            self._id = newid
            return self

    def clone(self, rekey=False):
        #a = copy.deepcopy(self)
        a = Activity.from_json(self.json(encode=False))
        if rekey:
            a.id(newid=shortuuid())
        return a
    
    def temporalpad(self, df):
        """Add a temporal pad of df=(before frames, after frames) or df=pad frames to the start and end of the activity.  The padded start frame may be negative."""
        df = (df, df) if not isinstance(df, tuple) else df
        self._startframe -= int(df[0])
        self._endframe += int(df[1])
        return self  

    def padto(self, t):
        """Add a symmetric temporal pad so that the activity is at least t seconds long"""
        return self.temporalpad(int(np.ceil(self.framerate()*((t - self.duration())/2.0)))) if t > self.duration() else self

    def disjoint(self, other, strict=False):
        """Enforce disjoint activities with other by shifting the endframe or startframe of self to not overlap if they share the same tracks.
           Other may be an Activity() or list of Activity()
           if strict=True, then throw an exception if other or self is fully contained with the other, resulting in degenerate activity after disjoint
        """
        for o in tolist(other):
            assert isinstance(o, Activity), "Invalid input - must be vipy.activity.Activity() or list of activities"       
            if strict:
                assert not (o.during(self.startframe()) and o.during(self.endframe())), "Self cannot fully overlap other"
                assert not (self.during(o.startframe()) and self.during(o.endframe())), "Other cannot fully overlap self"
            if o.trackids() == self.trackids() and o.during(self.endframe()):
                self.endframe(o.startframe()-1)
            if o.trackids() == self.trackids() and o.during(self.startframe()):
                self.startframe(o.endframe()+1)
        return self  # may be zero length now

    def temporal_distance(self, other):
        """Return the temporal distance in frames between self and other which is the minimum frame difference between the end of one to the start of the other, or zero if they overlap"""
        assert isinstance(other, Activity), "Invalid input - must be vipy.activity.Activity()"
        return (max(self.startframe(), other.startframe()) - min(self.endframe(), other.endframe())) if self.temporal_iou(other) == 0 else 0

    def within(self, startframe, endframe):
        """Is the activity within the frame rate (startframe, endframe)?"""
        return self.startframe() >= startframe and self.endframe() <= endframe
    
