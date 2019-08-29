import time
import numpy.random
from itertools import imap, count

class RandomVector(object):
    """a repeatable, random stream of vectors with coordinates uniformly distributed in [0,1]"""
    
    _seed = None
    _dimensionality = None
    _streamlength = None
    
    def __init__(self, dimensionality=128, streamlength=None):
        self._seed = time.time()
        self._dimensionality = dimensionality
        self._streamlength = streamlength

    def __iter__(self):
        numpy.random.seed(int(self._seed))
        f = lambda x: numpy.random.rand(self._dimensionality)
        if self._streamlength is not None:
            return imap(f, iter(range(self._streamlength)))            
        else:
            return imap(f, count())

class RandomBinaryVector(RandomVector):
    """a repeatable, random stream of binary vectors"""
    
    def __iter__(self):
        numpy.random.seed(int(self._seed))
        f = lambda x: numpy.random.rand(self._dimensionality) > 0.5
        if self._streamlength is not None:
            return imap(f, iter(range(self._streamlength)))            
        else:
            return imap(f, count())
        

class RandomLabeledBinaryVector(object):        
    """a repeatable, random stream of (binaryvector, integerlabel) tuples"""    
    _seed = None
    _dimensionality = None
    _streamlength = None
    _numlabels = None
    
    def __init__(self, dimensionality=128, streamlength=None, numlabels=8):
        self._seed = time.time()
        self._dimensionality = dimensionality
        self._streamlength = streamlength
        self._numlabels = numlabels        

    def __iter__(self):
        numpy.random.seed(int(self._seed))
        f = lambda x: (numpy.random.rand(self._dimensionality) > 0.5, numpy.random.random_integers(0, self._numlabels))
        if self._streamlength is not None:
            return imap(f, iter(range(self._streamlength)))  # bounded
        else:
            return imap(f, count())  # unbounded
    
