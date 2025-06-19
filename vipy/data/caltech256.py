import os
from vipy.downloader import download_and_unpack
from vipy.dataset import Dataset
from vipy.util import remkdir, tocache
from vipy.image import ImageCategory


URL = 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar'
SHA1 = '2195e9a478cf78bd23a1fe51f4dabe1c33744a1c'


class Caltech256(Dataset):
    """Caltech-256 dataset: https://data.caltech.edu/records/nyy15-4j048"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('caltech256') if datadir is None else datadir
        
        # Download (if not cached)
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            download_and_unpack(URL, self._datadir, sha1=SHA1)            
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, '256_ObjectCategories')        
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for imf in os.listdir(imdir):
                imlist.append((category.split('.')[1], os.path.join(categorydir, category, imf)))

        loader = lambda x, categorydir=categorydir: ImageCategory(filename=x[1], category=x[0])
        super().__init__(imlist, id='caltech-256', loader=loader)
            
        # Done
        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
