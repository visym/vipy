from vipy.geometry import BoundingBox

class Detection(BoundingBox):
    """Represent a single bounding box with a label and confidence for an object detection in a scene"""
    def __init__(self, label='object', xmin=None, ymin=None, width=None, height=None, xmax=None, ymax=None, confidence=None, xcentroid=None, ycentroid=None):
        super(Detection, self).__init__(xmin=xmin, ymin=ymin, width=width, height=height, xmax=xmax, ymax=ymax, xcentroid=xcentroid, ycentroid=ycentroid)
        self._label = str(label)
        self._confidence = float(confidence) if confidence is not None else confidence        
        
    def __repr__(self):
        strlist = []
        if self.category() is not None: 
            strlist.append('category="%s"' % self.category())
        if self.isvalid():
            strlist.append('bbox=(xmin=%1.1f,ymin=%1.1f,xmax=%1.1f,ymax=%1.1f)' %
                           (self.bbox.xmin(), self.bbox.ymin(),self.bbox.xmax(), self.bbox.ymax()))
        if self._confidence is not None:
            strlist.append('conf=%1.3f')
        return str('<vipy.object.detection: %s>' % (', '.join(strlist)))
            
    def __str__(self):
        return self.__repr__()

    def category(self):
        return self._label

    def label(self):
        return self._label

    


