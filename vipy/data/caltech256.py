import os
import vipy.downloader
import vipy.dataset
from vipy.util import remkdir
from vipy.image import ImageCategory


URL = 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar'
SHA1 = None


class Caltech256(vipy.dataset.Dataset):
    """Caltech-256 dataset: https://data.caltech.edu/records/nyy15-4j048"""
    def __init__(self, datadir):
        # Download (if not cached)
        self._datadir = remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, '256_ObjectCategories')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, '256_ObjectCategories')        
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        super().__init__(imlist, id='caltech-256')
            
