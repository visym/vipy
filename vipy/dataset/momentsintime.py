import os
import numpy as np
from vipy.util import filetail, remkdir, readjson, groupbyasdict, filefull, readlist, readcsv
from vipy.video import VideoCategory, Video, Scene
from vipy.object import Track, BoundingBox, Activity
from vipy.batch import Batch
import vipy.downloader
import vipy.visualize


class MultiMoments(object):
    def __init__(self, datadir):
        """Multi-Moments in Time:  http://moments.csail.mit.edu/"""
        self.datadir = datadir
        if not self._isdownloaded():
            raise ValueError('Not downloaded')
            
    def __repr__(self):
        return str('<vipy.dataset.multimoments: datadir="%s">' % (self.datadir, self.annodir))

    def _isdownloaded(self):
        return False
    
    def _dataset(self, csvfile):
        pass

    def categories(self):
        pass

    def trainset(self):
        pass

    def testset(self):
        pass

