"""Parses a dlib detection file for a dataset"""
import os
from csv import DictReader
from bobo.image import ImageDetection
from bobo.video import VideoDetection
import bobo.app
import numpy as np
import janus.visualize
from bobo.show import savefig
from bobo.util import filepath, tolist

BASE_SCHEMA = ['FILENAME', 'MEDIA_ID', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT',
               'LeftBrowLeftCorner_X', 'LeftBrowLeftCorner_Y', 'LeftBrowCenter_X', 'LeftBrowCenter_Y', 'LeftBrowRightCorner_X', 'LeftBrowRightCorner_Y',
               'RightBrowLeftCorner_X', 'RightBrowLeftCorner_Y', 'RightBrowCenter_X', 'RightBrowCenter_Y', 'RightBrowRightCorner_X', 'RightBrowRightCorner_Y',
               'LeftEyeLeftCorner_X', 'LeftEyeLeftCorner_Y', 'LeftEyeCenter_X', 'LeftEyeCenter_Y', 'LeftEyeRightCorner_X', 'LeftEyeRightCorner_Y',
               'RightEyeLeftCorner_X', 'RightEyeLeftCorner_Y', 'RightEyeCenter_X', 'RightEyeCenter_Y', 'RightEyeRightCorner_X', 'RightEyeRightCorner_Y',
               'LeftEar_X', 'LeftEar_Y', 'NoseLeft_X', 'NoseLeft_Y', 'NoseCenter_X', 'NoseCenter_Y', 'NoseRight_X', 'NoseRight_Y',
               'RightEar_X', 'RightEar_Y', 'MouthLeftCorner_X', 'MouthLeftCorner_Y', 'MouthCenter_X', 'MouthCenter_Y', 'MouthRightCorner_X', 'MouthRightCorner_Y',
               'ChinCenter_X', 'ChinCenter_Y']


TRACK_SCHEMA = BASE_SCHEMA[0:2] + ['TRACK_ID'] + BASE_SCHEMA[2:]
POSE_SCHEMA = TRACK_SCHEMA + ['PITCH', 'YAW', 'ROLL']

SCHEMAS = [BASE_SCHEMA, TRACK_SCHEMA, POSE_SCHEMA]


class DlibDetections(object):
    def __init__(self, csvfile, sparkContext=None):
        self.csvfile = csvfile
        self.datadir = filepath(csvfile)
        if not os.path.isfile(self.csvfile):
            raise ValueError("Can't find csv file at '%s'" % self.csvfile)
        self.sparkContext = sparkContext
        with open(self.csvfile, 'r') as f:
            try:
                header = next(f).strip().split(',')
                self.SCHEMA = [s for s in SCHEMAS if s == header][0]
            except IndexError:
                raise ValueError, 'Unrecognized schema [%r] in %s' % (header, self.csvfile)
            except StopIteration:
                print '[viset.dlib] Warning: csv file is empty: %s' % self.csvfile
                self.SCHEMA = BASE_SCHEMA

    def __repr__(self):
        return str('<viset.dlib: %s>' % self.csvfile)

    def rdd(self):
        self.sparkContext = bobo.app.init('viset_dlib') if self.sparkContext is None else self.sparkContext
        datadir = self.datadir

        xidx = self.SCHEMA.index('FACE_X')
        yidx = self.SCHEMA.index('FACE_Y')
        widx = self.SCHEMA.index('FACE_WIDTH')
        hidx = self.SCHEMA.index('FACE_HEIGHT')

        # Parse CSV file based on protocol, all non-detection properties go in attributes dictionary
        rdd = (self.sparkContext.textFile(self.csvfile)
                .map(lambda row: row.encode('utf-8').split(',')) # CSV to list of row elements
                .filter(lambda x: 'FILENAME' not in x[0])  # hack to ignore header
                .map(lambda x: ImageDetection(filename=os.path.join(datadir, x[0]), category='face',
                                              xmin = float(x[xidx]) if len(x[xidx]) > 0 else float('nan'),
                                              ymin = float(x[yidx]) if len(x[yidx]) > 0 else float('nan'),
                                              xmax = float(x[xidx])+float(x[widx]) if ((len(x[xidx])>0) and (len(x[widx])>0)) else float('nan'),
                                              ymax = float(x[yidx])+float(x[hidx]) if ((len(x[yidx])>0) and (len(x[hidx])>0)) else float('nan'),
                                              attributes={k:v for (k,v) in zip(self.SCHEMA,x)})))  # Parse row

        if 'TRACK_ID' in self.SCHEMA:
            rdd = (rdd.keyBy(lambda im: im.attributes['MEDIA_ID'] + im.attributes['TRACK_ID']) # MEDIA_ID+TRACK_ID is unique across videos
                      .reduceByKey(lambda a,b: tolist(a)+tolist(b))
                      .map(lambda (k,fr): VideoDetection(frames=fr, attributes={'MEDIA_ID':fr[0].attributes['MEDIA_ID'], 'TRACK_ID':fr[0].attributes['TRACK_ID']}) if len(tolist(fr))>1 else fr))

        return rdd

    def stream(self):
        datadir = self.datadir

        xidx = self.SCHEMA.index('FACE_X')
        yidx = self.SCHEMA.index('FACE_Y')
        widx = self.SCHEMA.index('FACE_WIDTH')
        hidx = self.SCHEMA.index('FACE_HEIGHT')

        with open(self.csvfile, 'r') as f:
            try:
                next(f)  #skip header
            except StopIteration:
                print '[viset.dlib] Warning: csv file is empty: %s' % self.csvfile
                return []

            imset = []
            for l in f:
                x = l.strip().split(',')
                im = ImageDetection(filename=os.path.join(datadir, x[0]), category='face',
                                    xmin = float(x[xidx]) if len(x[xidx]) > 0 else float('nan'),
                                    ymin = float(x[yidx]) if len(x[yidx]) > 0 else float('nan'),
                                    xmax = float(x[xidx])+float(x[widx]) if ((len(x[xidx])>0) and (len(x[widx])>0)) else float('nan'),
                                    ymax = float(x[yidx])+float(x[hidx]) if ((len(x[yidx])>0) and (len(x[hidx])>0)) else float('nan'),
                                    attributes={k:v for (k,v) in zip(self.SCHEMA,x)})  # Parse row
                imset.append(im)

        if 'TRACK_ID' in self.SCHEMA:
            framesets = {}
            sep = '|'
            for im in imset:
                key = im.attributes['MEDIA_ID'] + sep + im.attributes['TRACK_ID']
                if key not in framesets:
                    framesets[key] = []
                framesets[key].append(im)
            videos = []
            for k, fr in framesets.iteritems():
                media_id, track_id = k.split(sep)
                vid = VideoDetection(frames=fr, attributes={'MEDIA_ID': media_id, 'TRACK_ID': track_id})
                videos.append(vid)
            return videos

        return imset




def landmarks(im):
    """Return 21x2 frame array of landmark positions in 1-21 order, NaN if occluded"""
    lmidx = BASE_SCHEMA.index('LeftBrowLeftCorner_X')
    return np.float32(np.array([im.attributes[key] if len(im.attributes[key])>0 else np.float32('nan') for key in BASE_SCHEMA[lmidx:]])).reshape(21, 2)

def eyes_nose_chin(self, im):
    """Return 4x2 frame array of left eye, right eye nose chin"""
    fr = landmarks(im)
    return fr[[8-1, 11-1, 15-1, 21-1],:]  # left eye center, right eye center, nose center  (AFLW annotation, 1-indexed)

def show(im, figure=None):
    lm = landmarks(im)
    k_valid = np.argwhere(~np.isnan(lm).any(axis=1)).ravel().tolist()
    valid_landmarks = lm[k_valid]
    bbox = im.bbox.convexhull(valid_landmarks).maxsquare().dilate(1.1)
    imc = im.grayscale().crop(bbox).resize(rows=128)
    scale = float(128.0) / float(bbox.height())
    scaled_valid_landmarks = np.array([(scale*(x-bbox.xmin), scale*(y-bbox.ymin)) for (x,y) in valid_landmarks])
    bobo.show.imframe(imc.load(), scaled_valid_landmarks, 'b', markersize=30, label=None, figure=figure)

def montage(dlib):
    imlist = dlib.stream()
    for im in imlist:
        show(im, figure=1)
        im.filename(savefig())

    return janus.visualize.montage(imlist, 128,128, crop=False, grayscale=False)
