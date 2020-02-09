import os
from vipy.util import remkdir
import vipy.downloader
from vipy.image import ImageCategory

URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
SHA1 = 'b8ca4fe15bcd0921dfda882bd6052807e63b4c96'


class Caltech101(object):
    def __init__(self, datadir):
        """Caltech101, provide a datadir='/path/to/store/caltech101' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.caltech101: %s>' % self.datadir)

    def download_and_unpack(self):
        vipy.downloader.download_and_unpack(URL, self.datadir, sha1=SHA1)

    def dataset(self):
        # Return json or CSV file containing dataset description
        categorydir = os.path.join(self.datadir, '101_ObjectCategories')

        imlist = []
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        return imlist
