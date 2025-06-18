import os
import vipy


class PIP_370k(vipy.dataset.Dataset):
    URL = 'https://dl.dropboxusercontent.com/s/fai9ontpmx4xv9i/pip_370k.tar.gz'
    MD5 = '2cf844fbc78fde1c125aa250e99db19f'
    def __init__(self, datadir):
        jsonfile = os.path.join(datadir, 'pip_370k', 'pip_370k.json')        
        if not os.path.exists(jsonfile):
            vipy.downloader.download_and_unpack(PIP_370k.URL, datadir, md5=PIP_370k.MD5)
        super().__init__(vipy.load(jsonfile), id='pip_370k')

        
class PIP_175k(vipy.dataset.Dataset):
    URL = 'https://dl.dropboxusercontent.com/s/aqafx0t3k7691gc/pip_175k.tar.gz'
    MD5 = '2d8dce694e8a6056023c5232975297d9'
    def __init__(self, datadir):
        jsonfile = os.path.join(datadir, 'pip_175k', 'pip_175k.json')
        if not os.path.exists(jsonfile):
            vipy.downloader.download_and_unpack(PIP_175k.URL, datadir, md5=PIP_175k.MD5)
        super().__init__(vipy.load(jsonfile), id='pip_175k')
        
