import os
from vipy.util import remkdir
import vipy.downloader
import vipy.dataset
from vipy.image import ImageCategory


URL = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip'
SHA1 = 'd1cc0e3686b03d5e1a7e9b734c6d04f60857d674'


class Caltech101(vipy.dataset.Dataset):
    """Caltech-101 dataset: https://data.caltech.edu/records/mzrjq-6wc02"""
    def __init__(self, datadir):
        """Caltech101, provide a datadir='/path/to/store/caltech101' """

        # Download
        self._datadir = remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, 'caltech-101/101_ObjectCategories.tar.gz')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            
        if not os.path.exists(os.path.join(self._datadir, 'caltech-101/101_ObjectCategories/')):
            vipy.downloader.unpack(os.path.join(self._datadir, 'caltech-101/101_ObjectCategories.tar.gz'), os.path.join(self._datadir, 'caltech-101'))
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, 'caltech-101', '101_ObjectCategories')        
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        super().__init__(imlist, id='caltech-101')
            

        
