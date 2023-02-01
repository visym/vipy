import os
import vipy


URL = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
SHA1 = None


class Food101(vipy.dataset.Dataset):
    """Project: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"""
    def __init__(self, datadir):

        # Download
        self._datadir = vipy.util.remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, 'food-101.tar.gz')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            

        imlist = [vipy.image.ImageCategory(filename=f, category=vipy.util.filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(datadir, 'food-101'))]
        super().__init__(imlist, id='food101')

        





