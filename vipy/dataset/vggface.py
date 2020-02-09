import os
import numpy as np
from vipy.util import txtlist, dirlist, imlist, readcsv, filebase, remkdir, loadmat
from vipy.image import ImageDetection


class VGGFaceURL(object):
    def __init__(self, datadir='/proj/janus3/vgg-face/curated/vgg_face_dataset'):
        self.datadir = remkdir(datadir)
        self._subjects = None

    def __repr__(self):
        return str('<vipy.dataset.vggfaceurl: %s>' % self.datadir)

    def subjects(self):
        if self._subjects is None:
            self._subjects = [filebase(f) for f in txtlist(os.path.join(self.datadir, 'data'))]
        return self._subjects  # cached

    def dataset(self):
        """Return a generator to iterate over dataset"""
        SCHEMA = ['id', 'url', 'left', 'top', 'right', 'bottom', 'pose', 'detection_score', 'curation']
        for f in txtlist(os.path.join(self.datadir, 'files')):
            for r in readcsv(f, separator=' '):
                im = ImageDetection(url=r[2], category=filebase(f), xmin=float(r[3]), ymin=float(r[4]), xmax=float(r[5]), ymax=float(r[6]),attributes=dict(zip(SCHEMA,r)))
                yield im

    def take(self, n):
        """Randomly select n frames from dataset"""
        takelist = []
        SCHEMA = ['id', 'url', 'left', 'top', 'right', 'bottom', 'pose', 'detection_score', 'curation']
        for csvfile in np.random.choice(txtlist(os.path.join(self.datadir, 'data')), n):
            csv = readcsv(csvfile, separator=' ')
            r = csv[np.random.randint(1,len(csv))]  # not including header
            im = ImageDetection(url=r[2], category=filebase(csvfile), xmin=float(r[3]), ymin=float(r[4]), xmax=float(r[5]), ymax=float(r[6]), attributes=dict(zip(SCHEMA,r)))
            takelist.append(im)
        return takelist


class VGGFace(object):
    def __init__(self, datadir='/proj/janus3/vgg-face/vgg-face-janus'):
        self.datadir = datadir
        self._subjects = None

    def __repr__(self):
        return str('<vipy.dataset.vggface: %s>' % self.datadir)

    def subjects(self):
        if self._subjects is None:
            self._subjects = [filebase(d) for d in dirlist(os.path.join(self.datadir, 'images'))]
        return self._subjects  # cached

    def wordnetid_to_name(self):
        wnid = [str(x[0]) for x in loadmat('/proj/janus3/vgg-face/vgg-face-janus/vgg_face_wordnetid.mat')['name'][0]]
        name = [str(x[0][0]) for x in loadmat('/proj/janus3/vgg-face/vgg-face-janus/vgg_face_subjects.mat')['subjects']]
        return {k:v for (k,v) in zip(wnid, name)}

    def dataset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir, 'images')):
            for f in imlist(d):
                im = ImageDetection(filename=f, category=filebase(d))
                im = im.boundingbox(xmin=float(im.width() - 256) / 2.0, ymin=float(im.height() - 256.0) / 2.0, xmax=256.0 + ((im.width() - 256.0) / 2.0),ymax=256.0 + ((im.height() - 256.0) / 2.0))
                im = im.boundingbox(dilate=0.875)  # central 224x224
                yield im

    def fastset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir, 'images')):
            for f in imlist(d):
                im = ImageDetection(filename=f, category=filebase(d))
                yield im

    def take(self, n):
        S = np.random.choice(self.subjects(), n)
        takelist = []
        for d in dirlist(os.path.join(self.datadir, 'images')):
            if filebase(d) in S:
                f = np.random.choice(imlist(d),1)[0]
                im = ImageDetection(filename=f, category=filebase(d))
                im = im.boundingbox(xmin=float(im.width() - 256) / 2.0, ymin=float(im.height() - 256.0) / 2.0, xmax=256.0 + ((im.width() - 256.0) / 2.0),ymax=256.0 + ((im.height() - 256.0) / 2.0))
                im = im.boundingbox(dilate=0.875)  # central 224x224
                takelist.append(im)
        return takelist

    def by_subject(self, wordnetid):
        d = os.path.join(self.datadir, 'images', wordnetid)
        for f in imlist(d):
            yield ImageDetection(filename=f, category=filebase(d))
