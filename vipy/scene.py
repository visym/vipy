import vipy.show
from vipy.image import ImageCategory
from vipy.show import colorlist, savefig
from vipy.util import tolist, quietprint, imwrite, imread, tmpjpg
from copy import deepcopy
import numpy as np
import matplotlib.transforms


class Scene(ImageCategory):
    """A scene is an ImageCategory with one or more object detections"""
    def __init__(self, filename=None, url=None, category='scene', attributes=None, objects=None, array=None):
        super(Scene, self).__init__(filename=filename, url=url, attributes=attributes, category=category, array=array)   # ImageCategory class inheritance        
        self._objectlist = []
        self.filename(filename)  # override filename only        
        if filename is not None and objects is not None and len(objects) > 0:
            #self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self.filename(filename)  # override filename only
        elif url is not None and objects is not None and len(objects)>0:
            #self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self.url(url) # override url only
        else:
            super(Scene, self).__init__(filename=filename, url=url, attributes=attributes, category=category, array=array)   # ImageCategory class inheritance                   

        if objects is not None and len(objects)>0:
            #self.__dict__ = objects[0].__dict__.copy()  # shallow copy of all object attributes
            self._objectlist = objects
        self.category(category)
    
    def __repr__(self):
        str_size = ", height=%d, width=%d, color='%s'" % (self._array.shape[0], self._array.shape[1], 'gray' if self._array.ndim==2 else 'color') if self.isloaded() else ""
        str_file = ''
        str_category = "%scategory='%s'" % (', ' if len(str_file)>0 else '', self._category)
        str_objects = ", objects=%d" % len(self._objectlist)
        return str('<vipy.scenedetection: %s%s%s%s>' % (str_file, str_category, str_size, str_objects))

    def __len__(self):
        return len(self._objectlist)

    def __iter__(self):        
        for im in self._objectlist:
            yield im
    
    def __getitem__(self, k):        
        return self._objectlist[k]
    
    def append(self, imdet):
        self._objectlist.append(imdet)
        return self
    
    def show(self, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), outfile=None):
        """Show scene detection with an optional subset of categories"""
        #quietprint('[vipy.scenedetection][%s]: displaying scene' % (self.__repr__()), verbosity=2)                                            
        valid_categories = sorted(self.categories() if category is None else tolist(category))
        valid_detections = [im for im in self._objectlist if im.category() in valid_categories]        
        if categoryColor is None:
            colors = colorlist()
            categoryColor = dict([(c, colors[k]) for (k, c) in enumerate(valid_categories)])
        detection_color = [categoryColor[im.category()] for im in valid_detections]
        vipy.show.imdetection(self.rgb()._array, valid_detections, bboxcolor=detection_color, textcolor=detection_color, figure=figure, do_caption=do_caption, facealpha=boxalpha, fontsize=fontsize, captionlist=captionlist, captionoffset=captionoffset)
        if outfile is not None:
            savefig(outfile, figure)
        return self

    def savefig(self, outfile=None, category=None, figure=None, do_caption=True, fontsize=10, boxalpha=0.25, captionlist=None, categoryColor=None, captionoffset=(0,0), dpi=200):
        """Show a subset of object categores in current image and save to the given file"""
        outfile = outfile if outfile is not None else tmpjpg()
        self.show(category, figure, do_caption, fontsize, boxalpha, captionlist, categoryColor, captionoffset)
        savefig(outfile, figure, dpi=dpi, bbox_inches='tight', pad_inches=0)
        return outfile

    def objects(self, objectlist=None):
        if objectlist is None:
            return self._objectlist
        else:
            s = self.clone()
            s._objectlist = objectlist
            return s

    def clone(self):
        return deepcopy(self)
    
    def categories(self):
        return list(set([obj.category() for obj in self._objectlist]))
    
