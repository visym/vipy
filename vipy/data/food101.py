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
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, 'caltech-101', '101_ObjectCategories')        
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(vipy.image.ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        super().__init__(imlist, id='caltech-101')
            

        





