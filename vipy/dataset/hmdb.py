import os
from vipy.video import VideoCategory
from vipy.util import remkdir, filetail, isvideo, isinstalled
import vipy.downloader

URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
SHA1 = None


class HMDB(object):
    def __init__(self, datadir):
        """Human motion dataset, provide a datadir='/path/to/store/hmdb' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.hmdb: "%s">' % self.datadir)

    def download(self):
        vipy.downloader.download(URL, os.path.join(self.datadir, filetail(URL)))
        self._unpack(os.path.join(self.datadir, filetail(URL)), self.datadir)

    def dataset(self):
        """Return a list of VideoCategory objects"""
        vidlist = []
        for (idx_category, category) in enumerate(os.listdir(self.datadir)):
            if os.path.isdir(os.path.join(self.datadir, category)):
                for (idx_video, filename) in enumerate(os.listdir(os.path.join(self.datadir, category))):
                    if isvideo(filename):
                        vidlist.append(VideoCategory(filename=os.path.join(category, filename), category=category))
        return vidlist

    def _unpack(self, rarfile, outdir):
        """Require unrar on command line"""
        if not isinstalled('unrar'):
            raise ValueError('Unpacking requires the unrar utility on the command line')
        os.system('unrar e %s %s' % (rarfile, outdir))
        for (idx_category, rarfile) in enumerate(os.listdir(outdir)):
            (category, ext) = os.path.splitext(rarfile)
            if not os.path.isdir(os.path.join(outdir,category)):
                os.mkdir(os.path.join(outdir, category))
                os.mkdir(os.path.join(outdir, category, 'export'))
                cmd = 'unrar e %s %s' % (os.path.join(outdir,rarfile), os.path.join(outdir,category))
                os.system(cmd)
