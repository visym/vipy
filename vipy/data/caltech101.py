import os
from vipy.util import remkdir, tocache
from vipy.downloader import download_and_unpack, unpack
from vipy.dataset import Dataset
from vipy.image import ImageCategory


URL = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip'
SHA1 = 'd1cc0e3686b03d5e1a7e9b734c6d04f60857d674'


class Caltech101(Dataset):
    """Caltech-101 dataset: https://data.caltech.edu/records/mzrjq-6wc02"""
    def __init__(self, datadir=None, redownload=False):
        """Caltech101, provide a datadir='/path/to/store/caltech101' """
        datadir = tocache('caltech101') if datadir is None else datadir
        
        # Download        
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            download_and_unpack(URL, self._datadir, sha1=SHA1)            
            unpack(os.path.join(self._datadir, 'caltech-101/101_ObjectCategories.tar.gz'), os.path.join(self._datadir, 'caltech-101'))
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, 'caltech-101', '101_ObjectCategories')        
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for imf in os.listdir(imdir):
                imlist.append((category, os.path.join(categorydir, category, imf)))                

        loader = lambda x, categorydir=categorydir: ImageCategory(filename=x[1], category=x[0])                
        super().__init__(imlist, id='caltech-101', loader=loader)

        # Done
        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
