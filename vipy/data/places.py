import os
import vipy
from vipy.util import filebase

MD5 = ['67e186b496a84c929568076ed01a8aa1',
       '9b71c4993ad89d2d8bcbdc4aef38042f']
URL = ['http://data.csail.mit.edu/places-private/places365/train_large_places365standard.tar',
       'http://data.csail.mit.edu/places-private/places365/val_large.tar']


class Places356():
    """Project: http://places2.csail.mit.edu/download-private.html"""
    def __init__(self, datadir):
        self._datadir = vipy.util.remkdir(datadir)
        for (url, md5) in zip(URL, MD5):
            if not os.path.exists(os.path.join(self._datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, os.path.join(self._datadir, filebase(url)), md5=md5)


    def trainset(self):
        imlist = [vipy.image.ImageCategory(filename=f, category=filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(self._datadir, 'train_large_places365standard'))]
        return vipy.dataset.Dataset(imlist, id='places365_train')


    def valset(self):
        imlist = [vipy.image.ImageCategory(filename=f, category=filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(self._datadir, 'val_large'))]
        return vipy.dataset.Dataset(imlist, id='places365_val')

    
        
        




    
