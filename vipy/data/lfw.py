import os
import numpy as np
from vipy.util import remkdir, filetail, dirlist, imlist, readcsv, tocache
from vipy.image import ImageCategory
import vipy.downloader


URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
URL_NAMES = 'http://vis-www.cs.umass.edu/lfw/lfw-names.txt'
URL_PAIRS_DEV_TRAIN = 'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
URL_PAIRS_DEV_TEST = 'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'
URL_PAIRS_VIEW2 = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'


class LFW(vipy.dataset.Dataset):
    def __init__(self, datadir=None, redownload=False):
        """Datadir contains the unpacked contents of LFW from $URL -> /path/to/lfw"""

        datadir = tocache('lfw') if datadir is None else datadir
         
        self._datadir = remkdir(os.path.expanduser(datadir))

        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            self._download()

        loader = lambda x: ImageCategory(category=x[0], filename=x[1])
        
        super().__init__([(s, f) for s in self.subjects() for f in imlist(os.path.join(self._datadir, 'lfw', s))], id='lfw', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
    def _download(self, verbose=True):
        # Link may be dead ... 
        vipy.downloader.download_and_unpack(URL, self._datadir, verbose=verbose)
        return self

    def subjects(self):
        """List of all subject names"""
        return [filetail(d) for d in dirlist(os.path.join(self._datadir, 'lfw'))]

    def subject_images(self, subject):
        """List of Images of a subject"""
        fnames = imlist(os.path.join(self._datadir, 'lfw', subject))
        return [ImageCategory(category=subject, filename=f) for f in fnames]

    def _parse_pairs(self, txtfile):
        pairs = []
        for x in readcsv(os.path.join(self._datadir, 'lfw', txtfile), separator='\t'):
            if len(x) == 3:
                pairs.append((ImageCategory(category=x[0], filename=os.path.join(self._datadir, 'lfw', x[0], '%s_%04d.jpg' % (x[0], int(x[1])))),
                              ImageCategory(category=x[0], filename=os.path.join(self._datadir, 'lfw', x[0], '%s_%04d.jpg' % (x[0], int(x[2]))))))
            elif len(x) == 4:
                pairs.append((ImageCategory(category=x[0], filename=os.path.join(self._datadir, 'lfw', x[0], '%s_%04d.jpg' % (x[0], int(x[1])))),
                              ImageCategory(category=x[2], filename=os.path.join(self._datadir, 'lfw', x[2], '%s_%04d.jpg' % (x[2], int(x[3]))))))
            else:
                pass
        return pairs

    def _pairsDevTest(self):
        if not os.path.isfile(os.path.join(self._datadir, 'lfw', 'pairsDevTest.txt')):
            raise ValueError("Download and save text file to $datadir/pairsDevTest.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self._datadir, 'lfw' 'pairsDevTest.txt')))
        return self._parse_pairs('pairsDevTest.txt')

    def _pairsDevTrain(self):
        if not os.path.isfile(os.path.join(self._datadir, 'lfw', 'pairsDevTrain.txt')):
            raise ValueError("Download and save text file to $datadir/pairsDevTrain.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self._datadir, 'lfw', 'pairsDevTrain.txt')))
        return self._parse_pairs('pairsDevTrain.txt')

    def _pairs(self):
        if not os.path.isfile(os.path.join(self._datadir, 'lfw', 'pairs.txt')):
            raise ValueError("Download and save text file to $datadir/pairs.txt with 'wget %s -O %s'" % (URL_PAIRS_DEV_TRAIN, os.path.join(self._datadir, 'lfw', 'pairs.txt')))
        return self._parse_pairs('pairs.txt')
