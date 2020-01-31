from vipy.geometry import BoundingBox

class Detection(BoundingBox):
    """Represent a single bounding box with a label and confidence for an object detection in a scene"""
    def __init__(self, label, xmin=None, ymin=None, width=None, height=None, xmax=None, ymax=None, confidence=None):
        super(Detection, self).__init__(xmin=xmin, ymin=ymin, width=width, height=height, xmax=xmax, ymax=ymax)
        self._label = str(label)
        self._confidence = float(confidence) if confidence is not None else confidence        
        
    def __repr__(self):
        if self._confidence is None:
            return str('<vipy.object.Detection: label=%s, xmin=%s, ymin=%s, width=%s, height=%s>'% (self._label, str(self.xmin), str(self.ymin), str(self.height()), str(self.width())))
        else:
            return str('<vipy.object.Detection: label=%s, conf=%1.3f, xmin=%s, ymin=%s, width=%s, height=%s>'% (self._label, self._confidence, str(self.xmin), str(self.ymin), str(self.height()), str(self.width())))            

    def __str__(self):
        return self.__repr__()

    def category(self):
        return self._label

    def label(self):
        return self._label

    


