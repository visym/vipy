import os
import vipy


URL = 'https://objectnet.dev/downloads/objectnet-1.0.zip'


class Objectnet(vipy.dataset.Dataset):
    """Project: https://objectnet.dev, password set on website, must be bytes encoded (e.g. passwd=b'thepassword')"""
    def __init__(self, datadir=None, passwd='objectnetisatestset', url=URL, name='objectnet', redownload=False):

        datadir = tocache('objectnet') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(url, self._datadir, md5=None, passwd=passwd)

        imlist = vipy.util.findimages(os.path.join(datadir, 'objectnet-1.0'))
        loader = lambda f: vipy.image.ImageCategory(filename=f, category=vipy.util.filebase(vipy.util.filepath(f)))
        super().__init__(imlist, id=name, loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
