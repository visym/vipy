import os
import vipy


URL = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
SHA1 = 'ed21dfefc61fbe39294b9441739f7ac91b343882'


class Food101(vipy.dataset.Dataset):
    """Project: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('food101') if datadir is None else datadir        

        # Download
        self._datadir = vipy.util.remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            

        loader = lambda f: vipy.image.ImageCategory(filename=f, category=vipy.util.filebase(vipy.util.filepath(f)))
        imlist = vipy.util.findimages(os.path.join(datadir, 'food-101'))
        super().__init__(imlist, id='food101', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()





