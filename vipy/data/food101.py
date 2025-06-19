import os
from vipy.util import filebase, filepath, remkdir, findimages
from vipy.downloader import download_and_unpack, unpack
from vipy.dataset import Dataset
from vipy.image import ImageCategory


URL = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
SHA1 = 'ed21dfefc61fbe39294b9441739f7ac91b343882'


class Food101(Dataset):
    """Project: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('food101') if datadir is None else datadir        

        # Download
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            download_and_unpack(URL, self._datadir, sha1=SHA1)            

        loader = lambda f: ImageCategory(filename=f, category=filebase(filepath(f)))
        imlist = findimages(os.path.join(datadir, 'food-101'))
        super().__init__(imlist, id='food101', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()





