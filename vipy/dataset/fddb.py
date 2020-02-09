import os
from vipy.util import readcsv
from vipy.image import Scene
from vipy.object import Detection

URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
FOLDS_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
FOLDS_SHA1 = '94ce19ba3348425dfbb8bcbe55802490f8f152f8'


class FDDB(object):
    """Manages the FDDB dataset: http://vis-www.cs.umass.edu/fddb"""

    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.folds_dir = os.path.join(self.rootdir, 'FDDB-folds')

        if not os.path.isdir(os.path.join(self.rootdir, '2002')):
            raise ValueError('Download FDDB dataset manually, with "wget %s -O %s; cd %s; tar zxvf %s"' % (URL, os.path.join(self.rootdir, 'originalPics.tar.gz'), self.rootdir, 'originalPics.tar.gz'))

        if not os.path.isfile(os.path.join(self.folds_dir, 'FDDB-fold-01.txt')):
            raise ValueError('Download FDDB-folds dataset manually, with "wget %s -O %s; cd %s; tar zxvf %s"' % (FOLDS_URL, os.path.join(self.rootdir, 'FDDB-folds.tgz'), self.rootdir, 'FDDB-folds.tgz'))

    def __repr__(self):
        return str("<vipy.dataset.fddb: '%s'>" % self.rootdir)

    def fold(self, foldnum=1):
        """Return the foldnum as a list of vipy.image.Scene objects, each containing all vipy.object.Detection faces in the current image"""
        # fold_file = os.path.join(self.folds_dir, 'FDDB-fold-%02d.txt' % foldnum)
        k = 0
        rows = readcsv(os.path.join(self.folds_dir, 'FDDB-fold-%02d-ellipseList.txt' % foldnum), separator=' ')
        imscenes = []
        while k < len(rows):
            filename = rows[k][0]
            num_faces = int(rows[k + 1][0])
            bbox = [rows[j] for j in range(k + 2, k + 2 + num_faces)]
            k = k + 2 + len(bbox)

            # This ignores the rotation
            ims = Scene(filename=os.path.join(self.rootdir, '%s.jpg' % filename), objects=[Detection('face', xcentroid=bb[3], ycentroid=bb[4], width=2 * float(bb[1]), height=2 * float(bb[0])) for bb in bbox])
            imscenes.append(ims)
        return imscenes
