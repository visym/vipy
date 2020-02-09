import os
import numpy as np
from vipy.util import dirlist, imlist, readcsv, filebase, readlist, groupbyasdict
from vipy.image import ImageDetection


class VGGFace2(object):
    def __init__(self, datadir, seed=None):
        assert os.path.isdir(os.path.join(datadir, 'n000001')) and os.path.exists(os.path.join(datadir, 'identity_meta.csv')), 'Download and unpack VGGFace2 data and metadata (http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) to "%s"' % datadir

        self.datadir = datadir
        self._subjects = None
        if seed is not None:
            np.random.seed(seed)  # for repeatable np.random

    def __repr__(self):
        return str('<vipy.dataset.vggface2: %s>' % self.datadir)

    def subjects(self):
        if self._subjects is None:
            self._subjects = [filebase(d) for d in dirlist(self.datadir)]
        return self._subjects  # cached

    def wordnetid_to_name(self):
        csv = readcsv(os.path.join(self.datadir, 'identity_meta.csv'), ignoreheader=True)
        return {str(x[0]):str(x[1]).replace('"', '') for x in csv}

    def vggface2_to_vggface1(self):
        assert os.path.exists(os.path.join(self.datadir, 'class_overlap_vgg1_2.txt')), 'Download class_overlap_vgg1_2.txt to "%s"' % self.datadir
        csv = readcsv(os.path.join(self.datadir, 'class_overlap_vgg1_2.txt'), separator=' ', ignoreheader=True)
        return {x[0]:x[1] for x in csv}

    def name_to_wordnetid(self):
        d = self.wordnetid_to_name()
        return {v:k for (k,v) in d.items()}

    def names(self):
        return list(self.wordnetid_to_name().values())

    def trainset(self):
        assert os.path.exists(os.path.join(self.datadir, 'train_list.txt')), 'Download "train_list.txt" from http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ to "%s"' % self.datadir
        csv = readlist(os.path.join(self.datadir, 'train_list.txt'))
        return [os.path.join(self.datadir, x).strip() for x in csv]

    def testset(self):
        assert os.path.exists(os.path.join(self.datadir, 'test_list.txt')), 'Download "test_list.txt" from http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ to "%s"' % self.datadir
        csv = readlist(os.path.join(self.datadir, 'test_list.txt'))
        return [os.path.join(self.datadir, x).strip() for x in csv]

    def split(self, f):
        """Convert absolute path /path/to/subjectid/filename.jpg from training or testing set to (subjectid, filename.jpg)"""
        x = os.path.split(f)
        subjectid = os.path.split(x[-2])[-1]
        imagefile = x[-1]
        return (subjectid, imagefile)

    def frontalset(self, n_frontal=1):
        # http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_posetemp_imglist.txt
        assert(n_frontal >= 1 and n_frontal <= 10)
        assert os.path.exists(os.path.join(self.datadir, 'test_posetemp_imglist.txt')), 'Download "test_posetemp_imglist.txt" from (http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_posetemp_imglist.txt) to "%s"' % self.datadir
        d = groupbyasdict([x.strip().split('/') for x in readlist(os.path.join(self.datadir, 'test_posetemp_imglist.txt'))], lambda v: v[0])
        d_subjectid_to_frontallist = {k:[os.path.join(self.datadir, k, y[1]) for y in v[0:n_frontal]] for (k,v) in d.items()}  # first and second set of five are frontal
        for (k,v) in d_subjectid_to_frontallist.items():
            for f in v:
                yield ImageDetection(filename=f).category(k)

    def dataset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir)):
            for f in imlist(d):
                yield ImageDetection(filename=f).category(filebase(d))

    def fastset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir)):
            for f in imlist(d):
                yield ImageDetection(filename=f, category=filebase(d))

    def take(self, n, wordnetid=None):
        """Randomly select n images from the dataset, or n images of a given subjectid"""
        subjectid = np.random.choice(self.subjects(), n) if wordnetid is None else [wordnetid] * n
        takelist = []
        for s in subjectid:
            d = os.path.join(self.datadir, s)
            f = np.random.choice(imlist(d),1)[0]
            im = ImageDetection(filename=f).category(filebase(d))
            takelist.append(im)
        return takelist

    def take_per_subject(self, n):
        """Randomly select n images per subject from the dataset"""
        subjectid = self.subjects()
        takelist = []
        for s in subjectid:
            d = os.path.join(self.datadir, s)
            for k in range(0,n):
                f = np.random.choice(imlist(d),1)[0]
                im = ImageDetection(filename=f).category(filebase(d))
                takelist.append(im)
        return takelist

    def subjectset(self, wordnetid):
        """Iterator for single subject"""
        assert wordnetid in self.wordnetid_to_name().keys(), 'Invalid wordnetid "%s"' % wordnetid
        d = os.path.join(self.datadir, wordnetid)
        for f in imlist(d):
            yield ImageDetection(filename=f, category=filebase(d))
