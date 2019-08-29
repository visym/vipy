from strpy.bobo.image import ImageCategory
from strpy.bobo.show import imdetection, colorlist, savefig
from strpy.bobo.util import tolist, quietprint, imwrite, imread
from copy import deepcopy
from strpy.bobo.cache import Cache, CacheError
import numpy as np
import matplotlib.transforms

class SceneDetection(ImageCategory):

    def __init__(self, filename=None, url=None, category='scene', ignore=False, fetch=True, attributes=None, objects=None):
        super(SceneDetection, self).__init__(filename=filename, url=url, ignore=ignore, fetch=fetch, attributes=attributes, category=category)   # ImageCategory class inheritance        
        self.objectlist = []
        self.filename(filename)  # override filename only        
        if filename is not None and objects is not None and len(objects) > 0:
            self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self.filename(filename)  # override filename only
        elif url is not None and objects is not None and len(objects)>0:
            self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self.url(url) # override url only
        else:
            super(SceneDetection, self).__init__(filename=filename, url=url, ignore=ignore, fetch=fetch, attributes=attributes, category=category)   # ImageCategory class inheritance                   

        if objects is not None and len(objects)>0:
            #self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self.objectlist = objects
        self.category(category)
    
    def __repr__(self):
        str_size = ", height=%d, width=%d, color='%s'" % (self.data.shape[0], self.data.shape[1], 'gray' if self.data.ndim==2 else 'color') if self.isloaded() else ""
        if self.isvalid():
            str_file = "filename='%s'" % self._filename
        elif self.url() is not None:
            str_file = "url='%s'" % self.url()
        else:
            str_file = ''            
        str_category = "%scategory='%s'" % (', ' if len(str_file)>0 else '', self._category)
        str_objects = ", objects=%d" % len(self.objectlist)
        return str('<strpy.scenedetection: %s%s%s%s>' % (str_file, str_category, str_size, str_objects))

    def __len__(self):
        return len(self.objectlist)

    def __iter__(self):        
        for im in self.objectlist:
            yield im
    
    def __getitem__(self, k):        
        return self.objectlist[k]
    
    def append(self, imdet):
        self.objectlist.append(imdet)
        return self
    
    def show(self, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), outfile=None):
        """Show a subset of object categores in current image"""
        quietprint('[strpy.scenedetection][%s]: displaying scene' % (self.__repr__()), verbosity=2)                                            
        valid_categories = sorted(self.categories() if category is None else tolist(category))
        valid_detections = [im for im in self.objectlist if im.category() in valid_categories]        
        if categoryColor is None:
            colors = colorlist()
            categoryColor = dict([(c, colors[k]) for (k, c) in enumerate(valid_categories)])
        detection_color = [categoryColor[im.category()] for im in valid_detections]
        imdetection(self.rgb().data, valid_detections, bboxcolor=detection_color, textcolor=detection_color, figure=figure, do_caption=do_caption, facealpha=boxalpha, fontsize=fontsize, captionlist=captionlist, captionoffset=captionoffset)
        if outfile is not None:
            savefig(outfile, figure)
        return self

    def savefig(self, outfile, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), dpi=200):
        """Show a subset of object categores in current image and save to the given file"""
        self.show(category, figure, do_caption, fontsize, boxalpha, captionlist, categoryColor, captionoffset)
        savefig(outfile, figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
        return self

    def objects(self, objectlist=None):
        if objectlist is None:
            return self.objectlist
        else:
            s = self.clone()
            s.objectlist = objectlist
            return s

    def clone(self):
        return deepcopy(self)
    
    def filter(self, f):
        self.objectlist = [im for im in self.objectlist if f(im)]
        return self

    def asdict(self, f):
        """f(im) returns (k,v) pair"""
        return {f(im) for im in self.objectlist}

    def distinct(self, f):
        return list(set([f(im) for im in self.objectlist]))

    def categories(self):
        return self.distinct(lambda im: im.category())
    
    def aslist(self, f):
        """f(im) return (v) singleton"""
        return [f(im) for im in self.objectlist]

    def overlap(self, bb):
        return [im.boundingbox().overlap(bb) for im in self.objectlist]

    def argmax(self, f):
        return self.objectlist[np.argmax(np.array([f(im) for im in self.objectlist]))]

    def argmin(self, f):
        return self.objectlist[np.argmin(np.array([f(im) for im in self.objectlist]))]
                
    def map(self, f):
        self.objectlist = [f(im) for im in self.objectlist]
        return self
    
    def sort(self, f):
        self.objectlist.sort(key=f)
        return self

    def deal(self, f, X):
        self.objectlist = [f(im,x) for (im,x) in zip(self.objectlist, X)]
        return self


