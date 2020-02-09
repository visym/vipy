import os
from vipy.util import remkdir, isjpg
from vipy.image import ImageDetection
import vipy.downloader


URL = 'http://www.vision.ee.ethz.ch/datasets_extra/ethz_shape_classes_v12.tgz'
SHA1 = 'ae9b8fad2d170e098e5126ea9181d0843505a84b'
SUBDIR = 'ETHZShapeClasses-V1.2'
LABELS = ['Applelogos','Bottles','Giraffes','Mugs','Swans']


class ETHZShapes(object):
    def __init__(self, datadir):
        """ETHZShapes, provide a datadir='/path/to/store/ethzshapes' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.ethzshapes: "%s">' % self.datadir)

    def download_and_unpack(self):
        vipy.downloader.download_and_unpack(URL, self.datadir, sha1=SHA1)

    def dataset(self):
        categorydir = LABELS
        imlist = []
        for (idx_category, category) in enumerate(categorydir):
            imdir = os.path.join(self.datadir, SUBDIR, category)
            for filename in os.listdir(imdir):
                if isjpg(filename) and not filename.startswith('.'):
                    # Write image
                    im = os.path.join(self.datadir, SUBDIR, category, filename)

                    # Write detections
                    gtfile = os.path.join(self.datadir, SUBDIR, category, os.path.splitext(os.path.basename(filename))[0] + '_' + category.lower() + '.groundtruth')
                    if not os.path.isfile(gtfile):
                        gtfile = os.path.join(self.datadir, SUBDIR, category, os.path.splitext(os.path.basename(filename))[0] + '_' + category.lower() + 's.groundtruth')  # plural hack
                    for line in open(gtfile,'r'):
                        if line.strip() == '':
                            continue
                        (xmin,ymin,xmax,ymax) = line.strip().split()
                        imlist.append(ImageDetection(filename=im, category=category, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

        return imlist
