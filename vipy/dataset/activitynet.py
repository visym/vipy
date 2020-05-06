import os
from vipy.util import filetail, remkdir, readjson, groupbyasdict
import vipy.downloader
from vipy.video import VideoCategory, Video
import numpy as np


# http://activity-net.org/download.html
URL = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json'


class ActivityNet(object):
    def __init__(self, datadir):
        """Activitynet, provide a datadir='/path/to/store/activitynet' """
        self._url = URL
        self.datadir = remkdir(datadir)
        if not self._isdownloaded():
            self.download()
        
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
    
    def categories(self):
        return set([v.category() for v in self.trainset()])

    def analysis(self):
        C = self.categories()
        d_category_to_trainsize = {k:len(v) for (k,v) in groupbyasdict(self.trainset(), lambda x: x.category()).items()}

        top10 = sorted([(k,v) for (k,v) in d_category_to_trainsize.items()], key=lambda x: x[1])[-10:]
        print('top10 categories by number of instances in training set:')
        print(top10)

        bottom10 = sorted([(k,v) for (k,v) in d_category_to_trainsize.items()], key=lambda x: x[1])[0:10]
        print('bottom-10 categories by number of instances in training set:')
        print(bottom10)
        
        return d_category_to_trainsize
