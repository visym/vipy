import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist
import uuid
import copy
from vipy.object import Track

try:
    import ujson as json  # faster
except ImportError:
    import json


ACTIVITY_GUID = int(uuid.uuid4().hex[0:8], 16)  

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
    def __init__(self, startframe, endframe, framerate=None, label=None, shortlabel=None, category=None, tracks=None, attributes=None, actorid=None, confidence=None):
        assert not (label is not None and category is not None), "Activity() Constructor requires either label or category kwargs, not both"
        assert startframe <= endframe, "Start frame must be less than or equal to end frame"
        if tracks:
            tracks = [v for (k,v) in tracks.items()] if isinstance(tracks, dict) else tracks  # backwards compatible, track dictionary input
            assert (all([isstring(t) for t in tracks]) or all([isinstance(t, Track) for t in tracks])), "Invalid track input"
            if all([isinstance(t, Track) for t in tracks]):
                assert all([any([t.during(f) for f in range(startframe, endframe)]) for t in tracks]), "All tracks must be be present in at least one frame when this activity occurs"
                tracks = [t.id() for t in tracks]  
        trackid = set(tracks) if tracks is not None else (set([actorid]) if actorid is not None else set([]))  # only store IDs, not tracks
        
        if tracks is not None and actorid is not None and actorid not in trackid:
            trackid.add(actorid)

        global ACTIVITY_GUID; self._id = hex(int(ACTIVITY_GUID))[2:];  ACTIVITY_GUID = ACTIVITY_GUID + 1;  # faster, increment package level UUID4 initialized GUID
        self._startframe = int(startframe)
        self._endframe = int(endframe)
        self._framerate = framerate
        self._label = category if category is not None else label        
        self._shortlabel = self._label if shortlabel is None else shortlabel
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
        return obj(startframe=int(d['_startframe']),
                   endframe=int(d['_endframe']),
                   framerate=d['_framerate'],
                   category=d['_label'],
                   shortlabel=d['_shortlabel'],
                   tracks=d['_trackid'],
                   attributes=d['attributes'],
                   actorid=d['_actorid'])
                
    def __len__(self):
        """Return activity length in frames, or zero if degenerate"""
        return max(0, self.endframe() - self.startframe())

    def __repr__(self):
        return str('<vipy.activity: category="%s", frames=(%d,%d), tracks=%s%s>' % (self.category(), self.startframe(), self.endframe(), len(self.trackids()), '' if self.confidence() is None else ', confidence=%1.2f' % self.confidence()))

    def dict(self):
        """Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding"""
        return self.json(s=None, encode=False)

    def json(self, encode=True):
        d = {k:v if k != '_trackid' else tuple(v) for (k,v) in self.__dict__.items()}
        return json.dumps(d) if encode else d
    
    def actorid(self, actorid=None):
        if actorid is None:
            return self._actorid
        else:
            self._actorid = actorid
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
        return int(np.round((self.endframe() - self.startframe()) / 2.0)) + self.startframe()

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

            fps = fps if fps is not None else (1.0/speed)*self._framerate            
            (self._startframe, self._endframe) = [int(np.round(f*(fps/float(self._framerate)))) for f in (self._startframe, self._endframe)]
            self._framerate = fps
            return self
    
    def category(self, label=None):
        if label is not None:
            self._label = label
            return self
        else:
            return self._label

    def label(self, label=None):
        """Alias for category"""
        return self.category(label)

    def shortlabel(self, label=None):
        """A optional shorter label string to show in the visualizations"""                
        if label is not None:
            self._shortlabel = label
            return self
        else:
            return self._shortlabel

    def add(self, track):
        """Add the track id for the track to this activity, so that if the track is changed externally it is reflected here"""
        assert isinstance(track, Track), "Invalid input - must be vipy.object.Track"
        assert track.during(self.startframe()) or track.during(self.endframe()) or track.during(self.middleframe()), "The track must be present during the activity"
        self._trackid.add(track.id())
        return self
        
    def tracks(self):
        """alias for trackids"""
        return self.trackids()

    def trackids(self):
        """Return a set of track IDs associated with this activity"""
        return set(self._trackid)

    def hasoverlap(self, other, threshold=0):
        assert isinstance(other, Activity), "Invalid input"
        assert threshold >= 0 and threshold <= 1, "Invalid temporal IOU threshold"
        return self.temporal_iou(other) > threshold
        
    def isneighbor(self, other, framegate=10):
        return self.temporal_iou(other.clone().temporalpad(framegate)) > 0 

    def hastrack(self, track):
        """Is the track part of the activity?"""
        assert isstring(track) or isinstance(track, Track), "Invalid input - Must be a vipy.object.Track().id() or vipy.object.Track()"
        trackid = track.id() if isinstance(track, Track) else track
        return trackid in self._trackid

    def append(self, newtrack):
        """Append newtrack to this activity and set as actorid()"""
        assert isinstance(newtrack, Track), "Invalid input - must be vipy.object.Track"
        self._trackid.add(newtrack.id())
        self.actorid(newtrack.id())
        return self

    def trackfilter(self, f):
        """Remove all tracks such that the lambda function f(trackid) resolves to False"""
        self._trackid = [tid for tid in self._trackid if f(tid)]
        if not f(self.actorid()):
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
            self._trackid.discard(oldtrackid)
            self._trackid.add(newtrackid)
            if self.actorid() == oldtrackid:
                self.actorid(newtrackid)
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
        assert self.actorid() == other.actorid(), "Actor ID must be the same"
        assert self.category() == other.category(), "Assigned activity is a different category"
        assert self.framerate() == other.framerate(), "Invalid input"
        assert confweight >= 0 and confweight <= 1, "Confidence weight must be [0,1]"

        self.startframe(min(other.startframe(), self.startframe()))
        self.endframe(max(other.endframe(), self.endframe()))
        if other.confidence() is not None and self.confidence() is not None:
            self.confidence(float((1.0-confweight)*self.confidence() + confweight*other.confidence()) if not maxconf else float(max(self.confidence(), other.confidence())))  # running mean confidence or max
        return self
        
    def temporal_iou(self, other):
        """Return the temporal intersection over union of two activities"""
        assert isinstance(other, Activity), "Invalid input - must be vipy.object.Activity()"
        assert self.framerate() == other.framerate(), "invald input - framerate must match"
        t_start = min(self.startframe(), other.startframe())
        t_end = max(self.endframe(), other.endframe())
        t_union = float(t_end - t_start)
        
        t_start = max(self.startframe(), other.startframe())
        t_end = min(self.endframe(), other.endframe())
        t_intersection = float(t_end - t_start)
        
        return (t_intersection / t_union) if t_intersection > 0 else 0
    
    def offset(self, dt):
        self._startframe = int(self._startframe + dt)
        self._endframe = int(self._endframe + dt)
        return self

    def truncate(self, startframe=None, endframe=None):
        """Truncate the activity so that it is between startframe and endframe"""
        self._startframe = self._startframe if startframe is None else max(self._startframe, startframe)
        self._endframe = self._endframe if endframe is None else min(self._endframe, endframe)
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
            global ACTIVITY_GUID; a.id(newid=hex(int(ACTIVITY_GUID))[2:]);  ACTIVITY_GUID = ACTIVITY_GUID + 1;  # faster, increment package level UUID4 initialized GUID
        return a
    
    def temporalpad(self, df):
        """Add a temporal pad of df=(before,after) or df=pad frames to the start and end of the activity.  The padded start frame may be negative."""
        df = (df, df) if not isinstance(df, tuple) else df
        self._startframe -= int(df[0])
        self._endframe += int(df[1])
        return self  

    def padto(self, t):
        """Add a symmetric temporal pad so that the activity is at least t seconds long"""
        return self.temporalpad((t - len(self))/2.0) if t > len(self) else self

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
