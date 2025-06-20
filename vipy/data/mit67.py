import os
from vipy.util import remkdir, tocache
from vipy.downloader import download_and_unpack
from vipy.dataset import Dataset
from vipy.image import ImageCategory


URL = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
SHA1 = '0b252516e746ba428b96af408d2e8162d9b08ac5'


class MIT67(Dataset):
    """IndoorSceneRecognition dataset: https://web.mit.edu/torralba/www/indoor.html"""
    def __init__(self, datadir=None, redownload=False):

        datadir = tocache('mit67') if datadir is None else datadir
        
        # Download
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            download_and_unpack(URL, self._datadir, sha1=SHA1)            
            
        # Create dataset
        imlist = []
        categorydir = os.path.join(self._datadir, 'Images')
        for category in os.listdir(categorydir):
            imdir = os.path.join(categorydir, category)
            for im in os.listdir(imdir):
                imlist.append( (os.path.join(categorydir, category, im), category) )

        loader = lambda x: ImageCategory(filename=x[0], category=x[1])
        super().__init__(imlist, id='mit67', loader=loader)
            
        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
