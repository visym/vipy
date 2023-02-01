import os
import vipy


URL = 'https://objectnet.dev/downloads/objectnet-1.0.zip'


class Objectnet(vipy.dataset.Dataset):
    """Project: https://objectnet.dev, password set on website, must be bytes encoded (e.g. passwd=b'thepassword')"""
    def __init__(self, datadir, passwd=None, url=URL, name='objectnet'):
        self._datadir = vipy.util.remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, vipy.util.filetail(url))):
            vipy.downloader.download_and_unpack(url, self._datadir, md5=None, passwd=passwd)

        imlist = [vipy.image.ImageCategory(filename=f, category=vipy.util.filebase(vipy.util.filepath(f))) for f in vipy.util.findimages(os.path.join(datadir, vipy.util.filebase(url)))]
        super().__init__(imlist, id=name)

