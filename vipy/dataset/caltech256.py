import os
from vipy.util import remkdir
import vipy.downloader
from vipy.image import ImageCategory

URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
SHA1 = '2195e9a478cf78bd23a1fe51f4dabe1c33744a1c'


class Caltech256(object):
    def __init__(self, datadir):
        """Caltech256, provide a datadir='/path/to/store/caltech256' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.caltech256: %s>' % self.datadir)

    def download_and_unpack(self):
        vipy.downloader.download_and_unpack(URL, self.datadir, sha1=SHA1)

    def dataset(self):
        # Return json or CSV file containing dataset description
        categorydir = os.path.join(self.datadir, '256_ObjectCategories')

        imlist = []
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        return imlist
