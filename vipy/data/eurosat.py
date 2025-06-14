import os
import vipy
from vipy.util import remkdir, filetail, filepath, tocache


URL = 'https://madm.dfki.de/files/sentinel/EuroSAT.zip'


class EuroSAT(vipy.dataset.Dataset):
    """https://github.com/phelber/EuroSAT"""
    def __init__(self, datadir=None, redownload=False):
        # Download (if not cached)
        datadir = tocache('d2d') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(URL, self._datadir, sha1=None)            

        # Create dataset
        imlist = tuple((f, filetail(filepath(f))) for (k,f) in enumerate(sorted(vipy.util.findimages(self._datadir))))
        loader = lambda x: vipy.image.ImageCategory(filename=x[0], category=x[1])
        super().__init__(imlist, id='eurosat', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
