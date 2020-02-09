import os
from vipy.util import filetail, remkdir, readjson
import vipy.downloader
from vipy.video import VideoCategory
import numpy as np


# http://activity-net.org/download.html
URL = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json'


class ActivityNet(object):
    def __init__(self, datadir):
        """ACtivitynet, provide a datadir='/path/to/store/activitynet' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.activitynet: "%s">' % self.datadir)

    def download(self):
        vipy.downloader.download(URL, os.path.join(self.datadir, filetail(URL)))
        return self

    def dataset(self):
        fps = 30.0  # is this right?
        jsonfile = os.path.join(self.datadir, filetail(URL))
        json = readjson(jsonfile)
        return [VideoCategory(url=v['url'],
                              filename=os.path.join(self.datadir, youtubeid),
                              category=a['label'],
                              startframe=int(np.round(a['segment'][0] * fps)),
                              endframe=int(np.round(a['segment'][1] * fps)))
                for (youtubeid, v) in json['database'].items() for a in v['annotations']]
