import os
from vipy.util import filetail, remkdir, readjson
import vipy.downloader
from vipy.video import VideoCategory, Video
import numpy as np


# http://activity-net.org/download.html
URL = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json'


class ActivityNet(object):
    def __init__(self, datadir):
        """Activitynet, provide a datadir='/path/to/store/activitynet' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.activitynet: "%s">' % self.datadir)

    def download(self):
        vipy.downloader.download(URL, os.path.join(self.datadir, filetail(URL)))
        return self

    def _isdownloaded(self):
        return os.path.exists(os.path.join(self.datadir, 'activity_net.v1-3.min.json'))
    
    def _dataset(self, subset):
        assert self._isdownloaded(), "Dataset not downloaded.  download() first or manually download '%s' into '%s'" % (self._url, self.datadir)        
        jsonfile = os.path.join(self.datadir, filetail(URL))
        json = readjson(jsonfile)
        return [VideoCategory(url=v['url'],
                              filename=os.path.join(self.datadir, youtubeid),
                              category=a['label'],
                              startsec=float(a['segment'][0]),
                              endsec=float(a['segment'][1]))
                for (youtubeid, v) in json['database'].items()
                for a in v['annotations']
                if v['subset'] == subset]

    def trainset(self):
        return self._dataset('training')

    def testset(self):
        """ActivityNet test set does not include any annotations"""
        assert self._isdownloaded(), "Dataset not downloaded.  download() first or manually download '%s' into '%s'" % (self._url, self.datadir)        
        json = readjson(os.path.join(self.datadir, filetail(URL)))
        return [Video(url=v['url'], filename=os.path.join(self.datadir, youtubeid)) for (youtubeid, v) in json['database'].items() if v['subset'] == 'testing']

    def valset(self):
        return self._dataset('validation')
    
    
