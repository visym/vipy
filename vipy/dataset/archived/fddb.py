import os
import csv
from bobo.cache import Cache
from bobo.util import remkdir, isstring
from bobo.image import ImageDetectionStream

URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
URL_ANNO = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
VISET = 'fddb'

cache = Cache()

