import os
from vipy.util import remkdir, tocache
import vipy.downloader
import vipy.dataset
from vipy.image import ImageCategory


URL = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
SHA1 = '0b252516e746ba428b96af408d2e8162d9b08ac5'


class MIT67(vipy.dataset.Dataset):
    """IndoorSceneRecognition dataset: https://web.mit.edu/torralba/www/indoor.html"""
    def __init__(self, datadir=tocache('mit67')):

        # Download
        self._datadir = remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, 'indoorCVPR_09.tar')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, 'Images')
        for category in os.listdir(categorydir):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append(ImageCategory(filename=os.path.join(categorydir, category, im), category=category))

        super().__init__(imlist, id='mit67')
            

        
