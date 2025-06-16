import os
from vipy.util import remkdir, tocache, filebase
import vipy.downloader
from vipy.dataset import Dataset
from vipy.image import ImageCategory


URL = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
SHA1 = '402d86b63cf3ace831f2af03bc9889e5e5c3dd1a'


class COIL100(Dataset):
    def __init__(self, datadir=None, redownload=False):

        datadir = tocache('coil100') if datadir is None else datadir
        
        # Download
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=SHA1)            
            
        # Create dataset
        imlist = []
        imgdir = os.path.join(self._datadir, 'coil-100')
        for f in os.listdir(imgdir):
            if '__' in f:
                imlist.append(f)

        loader = lambda f, imgdir=imgdir: ImageCategory(filename=os.path.join(imgdir, f), category=f.split('__')[0], attributes={'orientation':filebase(f).split('__')[1]})
        super().__init__(imlist, id='coil100', loader=loader)
            

        open(os.path.join(self._datadir, '.complete'), 'a').close()
        



