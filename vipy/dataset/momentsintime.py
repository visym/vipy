import os
import numpy as np
from vipy.util import filetail, remkdir, readjson, groupbyasdict, filefull, readlist, readcsv
from vipy.video import VideoCategory, Video, Scene
from vipy.object import Track, BoundingBox
from vipy.activity import Activity
import vipy.downloader
import vipy.visualize


class MultiMoments(object):
    def __init__(self, datadir):
        """Multi-Moments in Time:  http://moments.csail.mit.edu/
        
          >>> d = MultiMoments('/path/to/dir')
          >>> valset = d.valset()
          >>> valset.categories()           # return the dictionary mapping integer category to string
          >>> valset[1].categories()        # return set of categories for this clip
          >>> valset[1].category()          # return string encoded category for this clip (comma separated activity indexes)
          >>> valset[1].play()              # Play the original clip 
          >>> valset[1].mindim(224).show()  # Resize the clip to have minimum dimension 224, then show the modified clip
          >>> valset[1].centersquare().mindim(112).saveas('out.mp4')  # modify the clip as square crop from the center with mindim=112, and save to new file
          >>> valset[1].centersquare().mindim(112).normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).torch(startframe=0, length=16)  # export 16x3x112x112 tensor

        """
        self.datadir = datadir
        if not self._isdownloaded():
            raise ValueError('Not downloaded - datadir="/path/to/dir" such that "/path/to/dir/moments_categories.txt" exists.  See http://moments.csail.mit.edu/')
            
    def __repr__(self):
        return str('<vipy.dataset.multimoments: datadir="%s">' % (self.datadir))

    def _isdownloaded(self):
        return os.path.exists(os.path.join(self.datadir, 'moments_categories.txt'))
    
    def _dataset(self, csvfile):
        csv = readcsv(os.path.join(self.datadir, csvfile))
        categories = self.categories()        
        vidlist = []
        for r in csv:
            v = Scene(filename=os.path.join(self.datadir, 'videos', r[0]), category=','.join(sorted(r[1:])))
            for a in r[1:]:
                v.add(Activity(category=categories[int(a)], startframe=0, endframe=30*3))  # FIXME: framerate?
            vidlist.append(v)
        return vidlist

    def categories(self):
        return {int(r[1]):r[0] for r in readcsv(os.path.join(self.datadir, 'moments_categories.txt'))}

    def trainset(self):
        return self._dataset(os.path.join(self.datadir, 'trainingSet.txt'))  # FIXME: this is too slow due to ffmpeg-python hashing

    def valset(self):
        return self._dataset(os.path.join(self.datadir, 'validationSet.txt'))        

