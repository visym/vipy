import numpy as np
from vipy.geometry import BoundingBox
from vipy.util import isstring, tolist
import uuid
import copy
from vipy.object import Track


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
    def __init__(self, startframe, endframe, framerate=None, label=None, shortlabel=None, category=None, tracks=None, attributes=None, actorid=None):
        assert not (label is not None and category is not None), "Activity() Constructor requires either label or category kwargs, not both"
        assert startframe < endframe, "Start frame must be strictly less than end frame"
        if tracks:
            tracks = [v for (k,v) in tracks.items()] if isinstance(tracks, dict) else tracks  # backwards compatible, track dictionary input
            assert (all([isstring(t) for t in tracks]) or all([isinstance(t, Track) for t in tracks])), "Invalid track input"
            if all([isinstance(t, Track) for t in tracks]):
                assert all([any([t.during(f) for f in range(startframe, endframe)]) for t in tracks]), "All tracks must be be present in at least one frame when this activity occurs"
                tracks = [t.id() for t in tracks]  
               
        self._id = uuid.uuid1().hex
        self._startframe = startframe
        self._endframe = endframe
        self._framerate = framerate
        self._label = category if category is not None else label        
        self._shortlabel = self._label if shortlabel is None else shortlabel
        self._trackid = set(tracks) if tracks is not None else set([])
        self._actorid = actorid

        self.attributes = attributes if attributes is not None else {}            
        
    def __len__(self):
        """Return activity length in frames, or zero if degenerate"""
        return max(0, self.endframe() - self.startframe())

    def __repr__(self):
        return str('<vipy.activity: category="%s", frames=(%d,%d), tracks=%s>' % (self.category(), self.startframe(), self.endframe(), len(self.trackids())))

    def dict(self):
        return {'id':self._id, 'label':self.category(), 'shortlabel':self.shortlabel(), 'startframe':self._startframe, 'endframe':self._endframe, 'attributes':self.attributes, 'framerate':self._framerate,
                'trackid':self._trackid, 'actorid':self._actorid}
    
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
            self._endframe = f
            return self

    def middleframe(self):
        return int(np.round((self.endframe() - self.startframe()) / 2.0)) + self.startframe()

    def framerate(self, fps=None):
        """Resample (startframe, endframe) from known original framerate set by constructor to be new framerate fps"""        
        if fps is None:
            return self._framerate
        else:
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

    def hasoverlap(self, other):
        return self.temporal_iou(other) > 0
        
    def isneighbor(self, other, framegate=10):
        return self.temporal_iou(other.clone().temporalpad(framegate)) > 0 

    def hastrack(self, track):
        """Is the track part of the activity?"""
        assert isstring(track) or isinstance(track, Track), "Invalid input - Must be a vipy.object.Track().id() or vipy.object.Track()"
        trackid = track.id() if isinstance(track, Track) else track
        return trackid in self._trackid

    def replace(self, oldtrack, newtrack):
        """Replace oldtrack with newtrack if present in self._tracks.  Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change"""
        assert isinstance(oldtrack, Track) and isinstance(newtrack, Track), "Invalid input - must be vipy.object.Track"
        self._trackid.discard(oldtrack.id())
        self._trackid.add(newtrack.id())
        if self.actorid() == oldtrack.id():
            self.actorid(newtrack.id())
        return self    

    def replaceid(self, oldtrackid, newtrackid):
        """Replace oldtrack with newtrack if present in self._tracks.  Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change"""
        self._trackid.discard(oldtrackid)
        self._trackid.add(newtrackid)
        if self.actorid() == oldtrackid:
            self.actorid(newtrackid)
        return self    
    
    def during(self, frame):
        """Is frame during the time interval (startframe, endframe) inclusive?"""
        return int(frame) >= self._startframe and int(frame) <= self._endframe

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
        return self
    
    def id(self, newid=None):
        if newid is None:
            return self._id
        else:
            self._id = newid
            return self

    def clone(self):
        return copy.deepcopy(self)
    
    def temporalpad(self, df):
        """Add a temporal pad of df=(before,after) or df=pad frames to the start and end of the activity.  The padded start frame may be negative."""
        df = (df, df) if not isinstance(df, tuple) else df
        self._startframe -= int(df[0])
        self._endframe += int(df[1])
        return self  

    def disjoint(self, other, strict=False):
        """Enforce disjoint activities with other by shifting the endframe or startframe of self to not overlap.
           Other may be an Activity() or list of Activity()
           if strict=True, then throw an exception if other or self is fully contained with the other, resulting in degenerate activity after disjoint
        """
        for o in tolist(other):
            assert isinstance(o, Activity), "Invalid input - must be vipy.object.Activity() or list of activities"       
            if strict:
                assert not (o.during(self.startframe()) and o.during(self.endframe())), "Self cannot fully overlap other"
                assert not (self.during(o.startframe()) and self.during(o.endframe())), "Other cannot fully overlap self"
            if o.during(self.endframe()):
                self.endframe(o.startframe()-1)
            if o.during(self.startframe()):
                self.startframe(o.endframe()+1)
        return self  # may be zero length now

