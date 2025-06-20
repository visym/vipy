import os
from vipy.util import remkdir, readjson, groupbyasdict, tocache
import vipy.downloader
from vipy.video import VideoCategory
import numpy as np


class Kinetics700():
    """https://github.com/cvdfoundation/kinetics-dataset"""
    def __init__(self, datadir=tocache('kinetics700'), redownload=False):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(os.path.expanduser(datadir))
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics700.tar.gz'
        self._name = 'kinetics700'
        if redownload or not self.isdownloaded():
            self.download()
        
    def __repr__(self):
        return str('<vipy.data.%s: "%s/%s">' % (self._name, self.datadir, self._name))

    def download(self, verbose=True):
        vipy.downloader.download_and_unpack(self._url, self.datadir, verbose=verbose)
        return self

    def isdownloaded(self):
        return (os.path.exists(os.path.join(self.datadir, 'kinetics700.tar.gz')) or
                os.path.exists(os.path.join(self.datadir, self._name, 'train.json')))
    
    def _dataset(self, jsonfile, id=None):
        assert self.isdownloaded(), "Dataset not downloaded.  download() first or manually download '%s' to '%s' and unpack the tarball there" % (self._url, self.datadir)

        loader = lambda x: VideoCategory(url=x[0], filename=x[1], category=x[2], startsec=x[3], endsec=x[4])
        return vipy.dataset.Dataset([(v['url'],
                                      os.path.join(self.datadir, self._name, youtubeid),
                                      v['annotations']['label'],
                                      float(v['annotations']['segment'][0]),
                                      float(v['annotations']['segment'][1]))
                                     for (youtubeid, v) in readjson(jsonfile).items()], loader=loader, id=id)

    def trainset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'train.json'), self._name + ':train')

    def testset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'test.json'), self._name + ':test')

    def valset(self):
        return self._dataset(os.path.join(self.datadir, self._name, 'validate.json'), self._name + ':val')

    def categories(self):
        jsonfile = os.path.join(self.datadir, self._name, 'train.json')
        return set([v['annotations']['label'] for (youtubeid, v) in readjson(jsonfile).items()])
        
    def analysis(self):
        C = self.categories()
        d_category_to_trainsize = {k:len(v) for (k,v) in groupbyasdict(self.trainset(), lambda x: x.category()).items()}

        top10 = sorted([(k,v) for (k,v) in d_category_to_trainsize.items()], key=lambda x: x[1])[-10:]
        print('top10 categories by number of instances in training set:')
        print(top10)

        bottom10 = sorted([(k,v) for (k,v) in d_category_to_trainsize.items()], key=lambda x: x[1])[0:10]
        print('bottom-10 categories by number of instances in training set:')
        print(bottom10)
        
        return d_category_to_trainsize

        
class Kinetics600(Kinetics700):
    def __init__(self, datadir=tocache('kinetics600')):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(os.path.expanduser(datadir))
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz'
        self._name = 'kinetics600'


class Kinetics400(Kinetics700):
    def __init__(self, datadir=tocache('kinetics400')):
        """Kinetics, provide a datadir='/path/to/store/kinetics' """
        self.datadir = remkdir(os.path.expanduser(datadir))
        self._url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz'
        self._name = 'kinetics400'
