import numpy as np
import vipy.object


class Activity(object):
    def __init__(self, label, startframe, endframe, objects=None, attributes=None):
        self._startframe = startframe
        self._endframe = endframe
        self._label = label
        self._objectlist = objects
        if objects is not None:
            assert isinstance(objects, list) and all([isinstance(x, vipy.object.Detection) for x in objects]), "Invalid object list"
        self._attributes = attributes

    def __repr__(self):
        return str('<vipy.activity: category="%s", frames=(%d,%d), objects=%s>' % (self.category(), self.startframe(), self.endframe(), str(set([x.category() for x in self.objects()]))))

    def startframe(self):
        return self._startframe

    def endframe(self):
        return self._endframe

    def category(self, label=None):
        if label is not None:
            self._label = label
            return self
        else:
            return self._label

    def objects(self, objectlist=None):
        if objectlist is not None:
            self._objectlist = objectlist
            return self
        else:
            return self._objectlist

    def during(self, frame):
        return int(frame) >= self._startframe and int(frame) <= self._endframe

    def offset(self, dt):
        self._startframe = self._startframe - dt
        return self
