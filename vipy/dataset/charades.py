import os
from vipy.util import filetail, remkdir, readjson, groupbyasdict, filefull, readlist, readcsv
import vipy.downloader
from vipy.video import VideoCategory, Video, Scene
import numpy as np
from vipy.object import Track, BoundingBox, Activity


URL_ANNOTATIONS = 'http://ai2-website.s3.amazonaws.com/data/Charades.zip'
URL_DATA = 'http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip'


class Charades(object):
    def __init__(self, datadir, annodir):
        """Charades, provide paths such that datadir contains the contents of 'http://ai2-website.s3.amazonaws.com/data/Charades_v1.zip' and annodir contains 'http://ai2-website.s3.amazonaws.com/data/Charades.zip'"""
        self.datadir = datadir
        self.annodir = annodir
        if not self._isdownloaded():
            raise ValueError('Not downloaded')
            
    def __repr__(self):
        return str('<vipy.dataset.charades: datadir="%s", annotations="%s">' % (self.datadir, self.annodir))

    def _isdownloaded(self):
        return os.path.exists(os.path.join(self.annodir, 'Charades_v1_train.csv')) and os.path.exists(os.path.join(self.datadir, 'ZZXQF.mp4'))
    
    def _dataset(self, csvfile):
        csv = readcsv(csvfile)

        d_index_to_category = self.categories()
        vidlist = []
        for row in csv[1:]:            
            videoid = row[0]
            actions = row[-2]
            sceneloc = row[2]
            v = Scene(filename=os.path.join(self.datadir, '%s.mp4' % videoid), category=sceneloc, framerate=30.0)
            if len(actions) > 0:
                for a in actions.split(';'):
                    (category, startsec, endsec) = a.split(' ')
                    try:
                        v.add(Activity(category=d_index_to_category[category], startframe=float(startsec)*30, endframe=float(endsec)*30))
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print('[vipy.dataset.charades]: SKIPPING invalid activity row="%s" with error "%s"' % (str(row),str(e)))
            vidlist.append(v)
        return vidlist

    def categories(self):
        return {x.split(' ', 1)[0]:x.split(' ', 1)[1].strip() for x in readlist(os.path.join(self.annodir, 'Charades_v1_classes.txt'))}

    def trainset(self):
        return self._dataset(os.path.join(self.annodir, 'Charades_v1_train.csv'))

    def testset(self):
        return self._dataset(os.path.join(self.annodir, 'Charades_v1_test.csv'))

