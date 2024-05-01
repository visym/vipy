import os
import vipy


URL = ['http://data.csail.mit.edu/places/places365/train_256_places365standard.tar',
       'http://data.csail.mit.edu/places/places365/val_256.tar']
MD5 = [None, None]

class Places356():
    """Project: http://places2.csail.mit.edu/download-private.html"""
    def __init__(self, datadir=vipy.util.tocache('places365')):
        self._datadir = vipy.util.remkdir(datadir)
        for (url, md5) in zip(URL, MD5):
            if not os.path.exists(os.path.join(self._datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir, md5=md5)

    def trainset(self):
        imlist = [(f, vipy.util.filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(self._datadir, 'train_256_places365standard'))]
        loader = lambda x: vipy.image.ImageCategory(filename=x[0], category=x[1])
        return vipy.dataset.Dataset(imlist, id='places365_train', loader=loader)


    def valset(self):
        imlist = [(f, vipy.util.filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(self._datadir, 'val_256'))]
        loader = lambda x: vipy.image.ImageCategory(filename=x[0], category=x[1])        
        return vipy.dataset.Dataset(imlist, id='places365_val', loader=loader)

    
        
        




    
