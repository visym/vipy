import os
from vipy.util import readcsv, remkdir
from vipy.image import ImageDetection
import numpy as np


class FaceScrub(object):
    def __init__(self, datadir):
        self._datadir = datadir
        self._dataset = []  # parsed and validated ImageDetections

    def __repr__(self):
        return str('<vipy.dataset.facescrub: %s>' % self._datadir)

    def __len__(self):
        return len(self._dataset)

    def parse(self):
        """ Return a list of ImageDetections for all URLs in facescrub """
        imset = []
        imdir = remkdir(os.path.join(self._datadir, 'images'))
        csv_actors = readcsv(os.path.join(self._datadir, 'facescrub_actors.txt'), separator='\t')
        for (subjectname, imageid, faceid, url, bbox, sha256) in csv_actors[1:]:
            categoryname = subjectname.replace(' ', '_')
            (xmin,ymin,xmax,ymax) = bbox.split(',')
            imset.append(ImageDetection(url=url, filename=os.path.join(imdir, '%s_%s.jpg' % (categoryname, imageid)), category=categoryname, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, attributes={'GENDER':'male'}))

        csv_actresses = readcsv(os.path.join(self._datadir, 'facescrub_actresses.txt'), separator='\t')
        for (subjectname, imageid, faceid, url, bbox, sha256) in csv_actresses[1:]:
            categoryname = subjectname.replace(' ', '_')
            (xmin,ymin,xmax,ymax) = bbox.split(',')
            imset.append(ImageDetection(url=url, filename=os.path.join(imdir, '%s_%s.jpg' % (categoryname, imageid)), category=categoryname, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, attributes={'GENDER':'female'}))

        return imset

    def download(self):
        """ Download every URL in dataset and store in provided filename """
        return [d.download(ignoreErrors=True) for d in self.parse()]

    def validate(self):
        """Validate downloaded dataset and store cached list of valid bounding boxes and loadable images accessible with dataset()"""
        P = self.parse()
        D = []
        for (k,p) in enumerate(P):
            if k % 1000 == 0:
                print('[vipy.dataset.facescrub][%d/%d]: validating dataset... (successful download?, good bounding box?, loadable image?)' % (k, len(P)))
            if not p.invalid() and p.load(ignoreErrors=True, fetch=False) is not None:
                D.append(p.flush())
        self._dataset = D  # cache
        return self

    def dataset(self):
        return self._dataset if len(self) > 0 else self.validate().dataset()

    def stats(self):
        print('[vipy.dataset.facescrub]: %f percent downloaded' % (float(len(self.parse())) / float(len(self.dataset()))))

    def subjects(self):
        return list(set([im.category() for im in self.dataset()]))

    def split(self, valsize=128):
        D = self.dataset()
        subjects = list(set([im.category() for im in D]))
        (trainset, valset) = ([], [])
        for s in subjects:
            S = [d for d in D if d.category() == s]
            if len(S) > 3:
                valset = valset + S[0:2]   # two examples per subject
                trainset = trainset + S[2:]  # rest
            else:
                trainset = trainset + S
        valset = np.random.choice(valset, valsize)
        return (trainset, valset)
