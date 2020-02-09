import os
from vipy.util import remkdir, readjson
import vipy.downloader
from vipy.video import VideoCategory
import numpy as np


class Kinetics700(object):
    def __init__(self, datadir):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(datadir)
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics700.tar.gz'
        self._name = 'Kinetics700'

    def __repr__(self):
        return str('<vipy.dataset.%s: "%s/%s">' % (self._name, self.datadir, self._name))

    def download(self, verbose=True):
        vipy.downloader.download_and_unpack(self._url, self.datadir, verbose=verbose)
        return self

    def _dataset(self, jsonfile):
        fps = 30   # is thie correct?
        return [VideoCategory(url=v['url'],
                              filename=os.path.join(self.datadir, self._name, youtubeid),
                              category=v['annotations']['label'],
                              startframe=int(np.round(v['annotations']['segment'][0] * fps)),
                              endframe=int(np.round(v['annotations']['segment'][1] * fps)))
                for (youtubeid, v) in readjson(jsonfile).items()]

    def trainset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'train.json'))

    def testset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'test.json'))

    def valset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'validate.json'))


class Kinetics600(Kinetics700):
    def __init__(self, datadir):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(datadir)
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz'
        self._name = 'Kinetics600'


class Kinetics400(Kinetics700):
    def __init__(self, datadir):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(datadir)
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz'
        self._name = 'Kinetics400'
