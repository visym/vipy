import os
from vipy.dataset import Dataset
from vipy.video import VideoCategory
from vipy.util import remkdir, filetail, isvideo, isinstalled
import vipy.downloader
import re

URL = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
SHA1 = None


class UCF101(Dataset):
    def __init__(self, datadir):
        self.datadir = remkdir(datadir)

        if not os.path.exists(os.path.join(datadir, filetail(URL))):
            vipy.downloader.download(URL, os.path.join(self.datadir, filetail(URL)), tries=32)  # server fails many times, need smart continue
        if not len(vipy.util.videolist(datadir)) > 1:
            if not isinstalled('unrar'):
                raise ValueError('Unpacking requires the unrar utility on the command line.  On Ubuntu: "sudo apt install unrar", on macos "brew install rar"')
            os.system('unrar e %s %s' % (os.path.join(self.datadir, filetail(URL)), self.datadir))
            
        super().__init__([VideoCategory(filename=f, category=filetail(f).split('_')[1]).instanceid(f'ucf101:{k}') for (k,f) in enumerate(vipy.util.videolist(self.datadir))], 'ucf101')

    def as_space_separated_category(self):
        return self.map(lambda im: im.new_category(UCF101.to_space_separated_category(im.category())))
            
    @staticmethod
    def to_space_separated_category(c):
        """Convert CamelCase to a space separated phrase"""
        return ' '.join(re.findall(r'[A-Z][a-z]*', c))
