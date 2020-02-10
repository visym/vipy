import os
import numpy as np
from vipy.util import remkdir, filetail, dirlist, imlist, readcsv
from vipy.image import ImageCategory
import vipy.downloader


URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
URL_NAMES = 'http://vis-www.cs.umass.edu/lfw/lfw-names.txt'
URL_PAIRS_DEV_TRAIN = 'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
URL_PAIRS_DEV_TEST = 'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'
URL_PAIRS_VIEW2 = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'


class LFW(object):
    def __init__(self, datadir):
        """Datadir contains the unpacked contents of LFW from $URL -> /path/to/lfw"""
        self.lfwdir = remkdir(os.path.join(remkdir(datadir), 'lfw', 'lfw'))

    def download(self, verbose=True):
        vipy.downloader.download_and_unpack(URL, self.lfwdir, verbose=verbose)
        return self

    def __repr__(self):
        return str("<vipy.dataset.lfw: '%s'>" % self.lfwdir)

    def subjects(self):
        """List of all subject names"""
        return [filetail(d) for d in dirlist(self.lfwdir)]

    def subject_images(self, subject):
        """List of Images of a subject"""
        fnames = imlist(os.path.join(self.lfwdir, subject))
        return [ImageCategory(category=subject, filename=f) for f in fnames]

    def dataset(self):
        return [ImageCategory(category=s, filename=f) for s in self.subjects() for f in imlist(os.path.join(self.lfwdir, s))]

    def dictionary(self):
        """List of all Images of all subjects"""
        return {s:self.subject_images(s) for s in self.subjects()}

    def list(self):
        """List of all Images of all subjects"""
        subjectlist = []
        for (k,v) in self.dictionary().items():
            subjectlist = subjectlist + v
        return subjectlist

    def take(self, n=128):
        """Return a represenative list of 128 images"""
        return list(np.random.choice(self.list(), n))

    def _parse_pairs(self, txtfile):
        pairs = []
        for x in readcsv(os.path.join(self.lfwdir, txtfile), separator='\t'):
            if len(x) == 3:
                pairs.append((ImageCategory(category=x[0], filename=os.path.join(self.lfwdir, x[0], '%s_%04d.jpg' % (x[0], int(x[1])))),
                              ImageCategory(category=x[0], filename=os.path.join(self.lfwdir, x[0], '%s_%04d.jpg' % (x[0], int(x[2]))))))
            elif len(x) == 4:
                pairs.append((ImageCategory(category=x[0], filename=os.path.join(self.lfwdir, x[0], '%s_%04d.jpg' % (x[0], int(x[1])))),
                              ImageCategory(category=x[2], filename=os.path.join(self.lfwdir, x[2], '%s_%04d.jpg' % (x[2], int(x[3]))))))
            else:
                pass
        return pairs

    def pairsDevTest(self):
        if not os.path.isfile(os.path.join(self.lfwdir, 'pairsDevTest.txt')):
            raise ValueError("Download and save text file to $datadir/pairsDevTest.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self.lfwdir, 'pairsDevTest.txt')))
        return self._parse_pairs('pairsDevTest.txt')

    def pairsDevTrain(self):
        if not os.path.isfile(os.path.join(self.lfwdir, 'pairsDevTrain.txt')):
            raise ValueError("Download and save text file to $datadir/pairsDevTrain.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self.lfwdir, 'pairsDevTrain.txt')))
        return self._parse_pairs('pairsDevTrain.txt')

    def pairs(self):
        if not os.path.isfile(os.path.join(self.lfwdir, 'pairs.txt')):
            raise ValueError("Download and save text file to $datadir/pairs.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self.lfwdir, 'pairs.txt')))
        return self._parse_pairs('pairs.txt')
