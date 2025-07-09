import os
import vipy

        
class PIP_175k(vipy.dataset.Dataset):
    """https://visym.github.io/collector/pip_175k/"""
    URL = 'https://dl.dropboxusercontent.com/s/aqafx0t3k7691gc/pip_175k.tar.gz'
    MD5 = '2d8dce694e8a6056023c5232975297d9'
    def __init__(self, datadir):
        jsonfile = os.path.join(datadir, 'pip_175k', 'pip_175k.json')
        if not os.path.exists(jsonfile):

            vipy.downloader.download_and_unpack(PIP_175k.URL, datadir, md5=PIP_175k.MD5, tries=16)  # download fails repeatedly, keep trying

            if vipy.util.isinstalled('tar'):
                vipy.globals.log.info('extracting %s -> %s' % (os.path.join(datadir, vipy.util.filetail(PIP_175k.URL)), datadir))
                os.system('tar zxf %s --directory %s' % (os.path.join(datadir, vipy.util.filetail(PIP_175k.URL)), datadir))
            else:
                vipy.downloader.unpack(os.path.join(datadir, vipy.util.filetail(PIP_175k.URL)), datadir)

        super().__init__(vipy.load(jsonfile), id='pip_175k')

        
class PIP_370k_stabilized(vipy.dataset.Dataset):
    """https://visym.github.io/collector/pip_370k_stabilized/"""
    URL = 'https://dl.dropboxusercontent.com/s/fai9ontpmx4xv9i/pip_370k.tar.gz'
    MD5 = '2cf844fbc78fde1c125aa250e99db19f'
    def __init__(self, datadir):
        jsonfile = os.path.join(datadir, 'pip_370k', 'pip_370k.json')

        if not os.path.exists(jsonfile):        
            vipy.downloader.download_and_unpack(PIP_370k_stabilized.URL, datadir, md5=PIP_370k_stabilized.MD5, tries=16)  # download fails repeatedly, keep trying

            if vipy.util.isinstalled('tar'):                
                vipy.globals.log.info('extracting %s -> %s' % (os.path.join(datadir, vipy.util.filetail(PIP_370k_stabilized.URL)), datadir))
                os.system('tar zxf %s --directory %s' % (os.path.join(datadir, vipy.util.filetail(PIP_370k_stabilized.URL)), datadir))
            else:
                vipy.downloader.unpack(os.path.join(datadir, vipy.util.filetail(PIP_370k_stabilized.URL)), datadir)
                
        super().__init__(vipy.load(jsonfile), id='pip_370k')
