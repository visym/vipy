import os
import vipy.util
from vipy.util import filetail, remkdir, readjson, groupbyasdict, filefull, readlist, readcsv, filebase
import vipy.downloader
from vipy.video import VideoCategory, Video, Scene
import numpy as np
from vipy.object import Track, BoundingBox, Detection
from vipy.activity import Activity
from vipy.dataset import Dataset


# https://research.google.com/ava/download.html#ava_actions_download
URL = 'https://research.google.com/ava/download/ava_v2.2.zip'


class AVA(object):
    """https://research.google.com/ava/"""
    def __init__(self, datadir):
        """AVA, provide a datadir='/path/to/store/ava' """
        self.datadir = remkdir(datadir)
        if not self._isdownloaded():
            self.download()

            
    def __repr__(self):
        return str('<vipy.data.ava_v2.2: "%s">' % self.datadir)

    def download(self):
        zipfile = os.path.join(self.datadir, filetail(URL))
        vipy.downloader.download(URL, zipfile, verbose=False)
        vipy.downloader.unpack(zipfile, self.datadir, verbose=False)
        return self

    def _isdownloaded(self):
        return os.path.exists(os.path.join(self.datadir, 'ava_train_v2.2.csv'))
    
    def _dataset(self, csvfile, downloaded=False, verbose=False):
        # AVA csv format: video_id, middle_frame_timestamp, scaled_person_box (xmin, ymin, xmax, ymax), action_id, person_id

        # video_id: YouTube identifier
        # middle_frame_timestamp: in seconds from the start of the YouTube.
        # person_box: top-left (x1, y1) and bottom-right (x2,y2) normalized with respect to frame size, where (0.0, 0.0) corresponds to the top left, and (1.0, 1.0) corresponds to bottom right.
        # action_id: identifier of an action class, see ava_action_list_v2.2.pbtxt
        # person_id: a unique integer allowing this box to be linked to other boxes depicting the same person in adjacent frames of this video.
        
        assert self._isdownloaded(), "Dataset not downloaded.  download() first or manually download '%s' into '%s'" % (URL, self.datadir)
        csv = readcsv(csvfile)
        d_videoid_to_rows = groupbyasdict(csv, lambda x: x[0])

        vidlist = []
        d_category_to_index = self.categories()
        d_index_to_category = {v:k for (k,v) in d_category_to_index.items()}

        videos = [vipy.video.Scene(url='https://www.youtube.com/watch?v=%s' % video_id, framerate=None, 
                                   filename=os.path.join(self.datadir, video_id)) for (k_video, (video_id, rowlist)) in enumerate(d_videoid_to_rows.items())]                    
        for (k_video, (video_id, rowlist)) in enumerate(d_videoid_to_rows.items()):
            v = videos[k_video]

            if verbose:
                print('[vipy.data.ava][%d/%d]: Parsing "%s" with %d activities' % (k_video, len(d_videoid_to_rows), v.url(), len(rowlist)))            
            dummy_framerate = 30  # placeholder, don't know the true framerate until the video is downloaded

            # Tracks are "actor_id" across the video
            tracks = groupbyasdict(rowlist, lambda x: x[7])
            d_tracknum_to_track = {}

            for (k,(tracknum, tracklist)) in enumerate(tracks.items()):
                (keyframes, boxes) = zip(*[(((float(x[1]))*dummy_framerate), Detection(xmin=float(x[2]), ymin=float(x[3]), xmax=float(x[4]), ymax=float(x[5]), normalized_coordinates=True)) for x in tracklist])
                t = Track(keyframes=keyframes, boxes=boxes, category=tracknum, framerate=dummy_framerate, id=k)
                d_tracknum_to_track[tracknum] = t
                v.add_object(t, rangecheck=False)
                
            # Every row is a separate three second long activity centered at startsec involving one actor
            for (k,(video_id, startsec, xmin, ymin, xmax, ymax, activity_id, actor_id)) in enumerate(rowlist):
                t = d_tracknum_to_track[actor_id]
                try:
                    a = Activity(startframe=max(0, int(np.round(((float(startsec)-1.5)*dummy_framerate)))), endframe=int(np.round(((float(startsec)+1.5)*dummy_framerate))),
                                 category=d_index_to_category[int(activity_id)],
                                 tracks={t.id():t}, framerate=dummy_framerate, id=k)
                    v.add_object(a, rangecheck=False)

                except KeyboardInterrupt:
                    raise
                
                except Exception as e:
                    print('[vipy.data.ava]: actor_id=%s, activity_id=%s, video_id=%s - SKIPPING with error "%s"' % (actor_id, activity_id, video_id, str(e)))                   
                    raise

            start = float(max(0, (min([float(x[1]) for x in rowlist])-1.5)))
            end = float(max([float(x[1]) for x in rowlist])+1.5)
            v = v.clip(start, end)                
            vidlist.append(v)
        return vidlist

    def categories(self):
        rowlist = readlist(os.path.join(self.datadir, 'ava_action_list_v2.2.pbtxt'))
        rowlist = [r.strip() for r in rowlist]  # remove whitespace
        
        d_category_to_index = {}
        for (k,r) in enumerate(rowlist):
            if 'name' in r:

                category = str(r.replace('name: ', '').replace('"',''))
                index = int(rowlist[k+1].replace('label_id: ','').replace('"',''))
                d_category_to_index[category] = index
                
        return d_category_to_index

    def trainset(self):
        return Dataset(self._dataset(os.path.join(self.datadir, 'ava_train_v2.2.csv')), id='ava:train')

    def valset(self):
        return Dataset(self._dataset(os.path.join(self.datadir, 'ava_val_v2.2.csv')), id='ava:val')

    
