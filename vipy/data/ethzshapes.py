import os
import vipy
from vipy.util import remkdir, isjpg
from vipy.image import ImageDetection
import vipy.downloader


URL = 'https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/datasets/Dataset-information/ethz_shape_classes_v12.tgz'
SHA1 = 'ae9b8fad2d170e098e5126ea9181d0843505a84b'
SUBDIR = 'ETHZShapeClasses-V1.2'
LABELS = ['Applelogos','Bottles','Giraffes','Mugs','Swans']


class ETHZShapes(vipy.dataset.Dataset):
    def __init__(self, datadir=None, redownload=False):
        """ETHZShapes, provide a datadir='/path/to/store/ethzshapes' """

        datadir = tocache('ethzshapes') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)

        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)
        
        categorydir = LABELS
        imlist = []
        for (idx_category, category) in enumerate(categorydir):
            imdir = os.path.join(self._datadir, SUBDIR, category)
            for filename in os.listdir(imdir):
                if isjpg(filename) and not filename.startswith('.'):
                    # Write image
                    im = os.path.join(self._datadir, SUBDIR, category, filename)

                    # Write detections
                    gtfile = os.path.join(self._datadir, SUBDIR, category, os.path.splitext(os.path.basename(filename))[0] + '_' + category.lower() + '.groundtruth')
                    if not os.path.isfile(gtfile):
                        gtfile = os.path.join(self._datadir, SUBDIR, category, os.path.splitext(os.path.basename(filename))[0] + '_' + category.lower() + 's.groundtruth')  # plural hack
                    for line in open(gtfile,'r'):
                        if line.strip() == '':
                            continue
                        (xmin,ymin,xmax,ymax) = line.strip().split()
                        imlist.append( (im, category, xmin, ymin, xmax, ymax) )

        loader = lambda x: ImageDetection(filename=x[0], category=x[1], xmin=x[2], ymin=x[3], xmax=x[4], ymax=x[5])
        super().__init__(imlist, id='ethzshapes', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
