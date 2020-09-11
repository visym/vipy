import os
from vipy.util import filetail, remkdir, readjson, groupbyasdict, filefull, readlist, readcsv
import vipy.downloader
from vipy.video import VideoCategory, Video, Scene
import numpy as np
from vipy.object import Track, BoundingBox
from vipy.activity import Activity
import vipy.visualize
from vipy.batch import Batch

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
        d_videoid_to_video = {}
        for row in csv[1:]:            
            videoid = row[0]
            actions = row[-2]
            sceneloc = row[2]
            if videoid not in d_videoid_to_video:
                v = Scene(filename=os.path.join(self.datadir, '%s.mp4' % videoid), category=sceneloc)
                fps = v.probe()['streams'][0]['avg_frame_rate']
                fps = float(fps.split('/')[0]) / float(fps.split('/')[1])
                v.framerate(fps)  # FIXME: better handling of time based clips to avoid ffprobe
                d_videoid_to_video[videoid] = v                
            assert d_videoid_to_video[videoid].category() == sceneloc
            
            if len(actions) > 0:
                for a in actions.split(';'):
                    (category, startsec, endsec) = a.split(' ')
                    try:
                        d_videoid_to_video[videoid].add(Activity(category=d_index_to_category[category], startframe=float(startsec)*fps, endframe=float(endsec)*fps, attributes={'csvfile':row}))
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print('[vipy.dataset.charades]: SKIPPING invalid activity row="%s" with error "%s"' % (str(row),str(e)))
                        
        return list(d_videoid_to_video.values())

    def categories(self):
        return {x.split(' ', 1)[0]:x.split(' ', 1)[1].strip() for x in readlist(os.path.join(self.annodir, 'Charades_v1_classes.txt'))}

    def trainset(self):
        return self._dataset(os.path.join(self.annodir, 'Charades_v1_train.csv'))

    def testset(self):
        return self._dataset(os.path.join(self.annodir, 'Charades_v1_test.csv'))

    def review(self, outfile=None, mindim=1024, n=25):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in the train set"""
        T = self.trainset()
        quicklist = Batch(T).map(lambda v: [(c.load().quicklook(n=n), c.activitylist(), str(c.flush().print())) for c in v.mindim(512).activityclip()]).result()
        quicklooks = [imq for q in quicklist for (imq, activitylist, description) in q]  # for HTML display purposes
        provenance = [{'clip':str(description), 'activity':str(a), 'category':a.category(), 'train.csv':a.attributes['csvfile']} for q in quicklist for (imq, activitylist, description) in q for a in activitylist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='Charades trainset quicklooks', outfile=outfile, mindim=mindim)
        
