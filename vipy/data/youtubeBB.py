import os
from vipy.util import remkdir, tocache, readcsv, filetail, groupbyasdict, isinstalled
import vipy.downloader
from vipy.dataset import Dataset
from vipy.video import Scene
from vipy.object import Track, Detection


URLS = ['https://research.google.com/youtube-bb/yt_bb_classification_train.csv.gz',
        'https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz']


class YoutubeBB(Dataset):
    """https://research.google.com/youtube-bb/download.html

    Usage:

    To show the clips centered on a track at 30Hz
    >>> dataset = vipy.data.youtubeBB.YoutubeBB()
    >>> v = dataset.takeone().download()
    >>> for t in v.trackclip():
    >>>     t.show()

    To extract the first annotated keyframes at 1Hz:
    >>> imlist = [t.framerate(1).frame(0) for t in dataset.takeone().trackclip()]
    >>> imlist[0].show()

    To extract all annotated keyframes at 1Hz for each tracked object:
    >>> framelist = [[im for im in t.framerate(1).load()] for t in dataset.takeone().trackclip()]
    >>> vipy.visualize.montage([im.centersquare().annotate() for frames in framelist for im in frames]).show()

    The boxes are available as:
    >>> objectlist = [(o.category(), o.xywh()) for frames in framelist for im in frames for o in im.objects()]

    Notes:
      - Some videos with a native framerate different from 30Hz may generate non-pixel accurate frames due to rounding errors in the ffmpeg fps filter chain
      - The videos are loaded with a framerate of 30Hz.  Boxes are annotated at 1Hz.  Objects are linearly interpolated to export at 30Hz
        but this means that some frames that have interpolated boxes may no longer precisely fall on the object
      - To recover the precise frame annotated once per second, use v.frame(n).  For example, for the frame at timestamp 2000ms, use v.frame(int(30*(2000/1000)))
      - Try v.show(timestamp=True) to show the frame index overlay along with the boxes to see which frame is being displayed
      - Delay the framerate conversion as late in the filter chain as possible (e.g. after trackclip)

    """
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('youtubeBB') if datadir is None else datadir
        
        # Download        
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            if not isinstalled('gunzip'):
                raise ValueError('Downloading requires the gunzip utility on the command line.  On Ubuntu: "sudo apt install gunzip"')                
            
            for url in URLS:
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))
                os.system('gunzip %s' % (os.path.join(self._datadir, filetail(url))))
            open(os.path.join(self._datadir, '.complete'), 'a').close()
            
        csv = readcsv(os.path.join(self._datadir, 'yt_bb_detection_train.csv'))

        # https://research.google.com/youtube-bb/download.html
        #youtube_id - same as above.
        #timestamp_ms - same as above.
        #class_id - same as above.
        #class_name - same as above.
        #object_id - (integer) an identifier of the object in the video. (see note below)
        #object_presence - same as above.
        #xmin - (float) a [0.0, 1.0] number indicating the left-most location of the bounding box in coordinates relative to the frame size.
        #xmax - (float) a [0.0, 1.0] number indicating the right-most location of the bounding box in coordinates relative to the frame size.
        #ymin - (float) a [0.0, 1.0] number indicating the top-most location of the bounding box in coordinates relative to the frame size.
        #ymax - (float) a [0.0, 1.0] number indicating the bottom-most location of the bounding box in coordinates relative to the frame size.

        # Notes:
        # - (xmin, ymin, xmax, ymax) = (-1,-1,-1,-1) if object_presence == 'absent'
        # - All framerates are defined relative to 30Hz videos
        # - Keyframes are sampled once every second.  Linear interpolation of boxes may be noisy.  
        
        youtubeids = list(set([x[0] for x in csv]))
        d_youtubeid_to_objectids = {k:set(x[4] for x in v) for (k,v) in groupbyasdict(csv, lambda x: x[0]).items()}
        d_youtubeid_objectid_to_bboxes = {k:[(float(x[1]), x[3], (float(x[6]),float(x[8]),float(x[7]),float(x[9]))) for x in v] for (k,v) in groupbyasdict(csv, lambda x: (x[0], x[4])).items()}  # (timestamp_ms, class_name, ulbr)
        
        loader = (lambda ytid, d_youtubeid_to_objectids=d_youtubeid_to_objectids, d_youtubeid_objectid_to_bboxes=d_youtubeid_objectid_to_bboxes:
                  Scene(url='http://youtu.be/%s' % ytid, framerate=30.0,
                        tracks=[Track(category=d_youtubeid_objectid_to_bboxes[(ytid,o)][0][1],
                                      keyframes=[int(float(ts)*(30/1000)) for (ts, c, ulbr) in d_youtubeid_objectid_to_bboxes[(ytid,o)] if ulbr[0]>=0],
                                      boxes=[Detection(category=c, ulbr=ulbr, normalized_coordinates=True) for (ts, c, ulbr) in d_youtubeid_objectid_to_bboxes[(ytid,o)] if ulbr[0]>=0])
                                for o in d_youtubeid_to_objectids[ytid]]))
                  
        super().__init__(youtubeids, id='youtubeBB', loader=loader)


        
