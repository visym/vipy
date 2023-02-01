import os
import vipy


TRAIN_MD5 = '67e186b496a84c929568076ed01a8aa1'
TRAIN_URL = 'http://data.csail.mit.edu/places-private/places365/train_large_places365standard.tar'

VAL_URL = 'http://data.csail.mit.edu/places-private/places365/val_large.tar'
VAL_MD5 = '9b71c4993ad89d2d8bcbdc4aef38042f'


class Places356(vipy.dataset.Dataset):
    """Project: http://places2.csail.mit.edu/download-private.html"""
    def __init__(self, datadir, url=TRAIN_URL, md5=TRAIN_mD5, name='places365_train'):

        # Download
        self._datadir = vipy.util.remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, name, vipy.util.filetail(url))):
            vipy.downloader.download_and_unpack(url, os.path.join(self._datadir, name), md5=md5)            

        imlist = [vipy.image.ImageCategory(filename=f, category=vipy.util.filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(datadir, name))]
        super().__init__(imlist, id=name)

    def trainset(self):
        return self

    def valset(self):
        return Places365(self._datadir, VAL_URL, VAL_MD5, 'places365_val')

    
        
        




    
