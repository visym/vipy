import numpy as np
from vipy.geometry import BoundingBox


class Detection(BoundingBox):
    """Represent a single bounding box with a label and confidence for an object detection"""

    def __init__(self, label=None, xmin=None, ymin=None, width=None, height=None, xmax=None, ymax=None, confidence=None, xcentroid=None, ycentroid=None, category=None):
        super(Detection, self).__init__(xmin=xmin, ymin=ymin, width=width, height=height, xmax=xmax, ymax=ymax, xcentroid=xcentroid, ycentroid=ycentroid)
        assert not (label is not None and category is not None), "Constructor requires either label or category kwargs, not both"
        self._label = category if category is not None else label
        self._confidence = float(confidence) if confidence is not None else confidence

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
        return isinstance(other, Detection) and xywh() == other.xywh() and self.category() == other.category()

    def __str__(self):
        return self.__repr__()

    def category(self):
        return self._label

    def label(self):
        return self._label


class Track(object):
    """Represent many bounding boxes of an instance through time"""

    def __init__(self, label, frames, boxes, confidence=None, attributes=None):
        self._label = label
        self._frames = frames
        self._boxes = boxes
        assert all([isinstance(bb, BoundingBox) for bb in boxes]), "Bounding boxes must be vipy.geometry.BoundingBox objects"
        assert all([bb.isvalid() for bb in boxes]), "Invalid bounding boxes"

    def __repr__(self):
        strlist = []
        if self.category() is not None:
            strlist.append('category="%s"' % self.category())
        strlist.append('frame=[%d,%d]' % (self.startframe, self.endframe))
        strlist.append('obs=%d' % len(self._frames))
        return str('<vipy.object.track: %s>' % (', '.join(strlist)))

    def __getitem__(self, k):
        return self._interpolate(k)

    def __iter__(self):
        for k in range(self._startframe, self._endframe):
            yield (k,self._interpolate(k))

    def __len__(self):
        return len(self._frames)

    def keyframes(self):
        """Return keyframes where there are track observations"""
        return self._frames

    def startframe(self):
        return np.min(self._frames)

    def endframe(self):
        return np.max(self._frames)

    def _interpolate(self, k):
        """Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at observed frames, with repeated endpoints"""
        (xmin, ymin, width, height) = zip(*[bb.to_xywh() for bb in self._boxes])
        return Detection(label=self._label,
                         xmin=np.interp(k, self._frames, xmin),
                         ymin=np.interp(k, self._frames, ymin),
                         width=np.interp(k, self._frames, width),
                         height=np.interp(k, self._frames, height))

    def category(self, label=None):
        if label is not None:
            self._label = label
            return self
        else:
            return self._label

    def during(self, k):
        return k >= self.startframe() and k < self.endframe()

    def offset(self, dt=0, dx=0, dy=0):
        self._boxes = [bb.offset(dx, dy) for bb in self._boxes]
        self._frames = list(np.array(self._frames) + dt)
        return self

    def rescale(self, s):
        """Rescale track boxes by scale factor s"""
        self._boxes = [bb.rescale(s) for bb in self._boxes]
        return self

    def scale(self, s):
        """Alias for rescale"""
        return self.rescale(s)

    def scalex(self, sx):
        """Rescale track boxes by scale factor sx"""
        self._boxes = [bb.scalex(sx) for bb in self._boxes]
        return self

    def scaley(self, sy):
        """Rescale track boxes by scale factor sx"""
        self._boxes = [bb.scaley(sy) for bb in self._boxes]
        return self

    def dilate(self, s):
        """Dilate track boxes by scale factor s"""
        self._boxes = [bb.dilate(s) for bb in self._boxes]
        return self

    def rot90cw(self, H, W):
        """Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent"""
        self._boxes = [bb.rot90cw(H, W) for bb in self._boxes]
        return self

    def rot90ccw(self, H, W):
        """Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent"""
        self._boxes = [bb.rot90ccw(H, W) for bb in self._boxes]
        return self
