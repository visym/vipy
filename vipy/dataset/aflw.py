import os
import sqlite3
import csv
from vipy.util import remkdir
from vipy.image import ImageDetection
import numpy as np


SCHEMA = ['FILENAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT',
          'LeftBrowLeftCorner_X', 'LeftBrowLeftCorner_Y', 'LeftBrowCenter_X', 'LeftBrowCenter_Y', 'LeftBrowRightCorner_X', 'LeftBrowRightCorner_Y',
          'RightBrowLeftCorner_X', 'RightBrowLeftCorner_Y', 'RightBrowCenter_X', 'RightBrowCenter_Y', 'RightBrowRightCorner_X', 'RightBrowRightCorner_Y',
          'LeftEyeLeftCorner_X', 'LeftEyeLeftCorner_Y', 'LeftEyeCenter_X', 'LeftEyeCenter_Y', 'LeftEyeRightCorner_X', 'LeftEyeRightCorner_Y',
          'RightEyeLeftCorner_X', 'RightEyeLeftCorner_Y', 'RightEyeCenter_X', 'RightEyeCenter_Y', 'RightEyeRightCorner_X', 'RightEyeRightCorner_Y',
          'LeftEar_X', 'LeftEar_Y', 'NoseLeft_X', 'NoseLeft_Y', 'NoseCenter_X', 'NoseCenter_Y', 'NoseRight_X', 'NoseRight_Y',
          'RightEar_X', 'RightEar_Y', 'MouthLeftCorner_X', 'MouthLeftCorner_Y', 'MouthCenter_X', 'MouthCenter_Y', 'MouthRightCorner_X', 'MouthRightCorner_Y',
          'ChinCenter_X', 'ChinCenter_Y']

LANDMARKS_3D = np.array([[-57.0899, 37.2398, 47.1156],
                         [-39.2328, 43.6173, 59.8098],
                         [-16.9114, 36.602, 65.2735],
                         [15.6142, 36.602, 65.2664],
                         [38.5733, 42.9796, 60.1435],
                         [55.7927, 37.2398, 49.5446],
                         [-50.0746, 20.6582, 48.3683],
                         [-34.1307, 21.9337, 56.6652],
                         [-19.4624, 18.7449, 53.8703],
                         [19.4407, 18.7449, 53.8993],
                         [33.4713, 21.2959, 56.124],
                         [46.8642, 21.2959, 50.6618],
                         [-76.2226, -26.5357, -12.6367],
                         [-17.5491, -23.9847, 68.4396],
                         [0.308036, -15.0561, 90.3836],
                         [18.8029, -25.898, 65.7138],
                         [78.1142, -27.1735, -16.2555],
                         [-28.3909, -53.3214, 58.766],
                         [-0.329719, -54.5969, 72.568],
                         [28.3693, -53.9592, 58.5929],
                         [-0.967474, -96.051, 63.4542]])


class AFLW(object):
    def __init__(self, datadir):
        self.datadir = remkdir(datadir)
        if not os.path.isdir(os.path.join(self.datadir)):
            raise ValueError('Download AFLW dataset manually to "%s" ' % self.datadir)

    def __repr__(self):
        return str('<vipy.dataset.aflw: %s>' % self.datadir)

    def dataset(self):
        csvfile = os.path.join(self.datadir, 'aflw.csv')
        with open(csvfile, 'r') as f:
            for x in f.readline().split(','):
                if x[0][0] != '#':
                    im = ImageDetection(filename=os.path.join(self.datadir, x[0]), category='face',
                                        xmin=float(x[1]) if len(x[1]) > 0 else float('nan'),
                                        ymin=float(x[2]) if len(x[2]) > 0 else float('nan'),
                                        xmax=float(x[1]) + float(x[3]) if ((len(x[1]) > 0) and (len(x[3]) > 0)) else float('nan'),
                                        ymax=float(x[2]) + float(x[4]) if ((len(x[2]) > 0) and (len(x[4]) > 0)) else float('nan'),
                                        attributes={k:v for (k,v) in zip(SCHEMA,x)})  # Parse row
                    yield im

    def export(self):
        """Export sqlite database file to aflw.csv"""
        dbfile = os.path.join(self.datadir, 'data', 'aflw.sqlite')
        db = sqlite3.connect(dbfile)
        cursor = db.cursor()

        outfile = os.path.join(self.datadir, 'aflw.csv')
        with open(outfile, 'wb') as csvfile:
            f = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            f.writerow([s if k > 0 else '#' + s for (k,s) in enumerate(SCHEMA)])  # comment first line
            faceidQuery = cursor.execute('SELECT face_id FROM Faces').fetchall()
            for faceid in faceidQuery:
                faceid = faceid[0]
                fileidQuery = cursor.execute('SELECT file_id FROM Faces WHERE face_id = "%s"' % str(faceid))
                fileid = str(fileidQuery.fetchone()[0])

                imgDataQuery = cursor.execute('SELECT db_id,filepath,width,height FROM FaceImages where file_id = "%s"' % str(fileid))
                imgData = imgDataQuery.fetchall()

                facerect = cursor.execute('SELECT x,y,w,h FROM FaceRect WHERE face_id = "%s"' % str(faceid)).fetchone()

                ptsQuery = cursor.execute('SELECT descr,FeatureCoords.x,FeatureCoords.y FROM FeatureCoords,FeatureCoordTypes WHERE face_id = "%s" AND FeatureCoords.feature_id = FeatureCoordTypes.feature_id' % (str(faceid))).fetchall()
                annoDict = {}
                for pts in ptsQuery:
                    annoDict['%s_X' % str(pts[0])] = pts[1]
                    annoDict['%s_Y' % str(pts[0])] = pts[2]

                annoDict['FILENAME'] = os.path.join('data','flickr', imgData[0][1])
                annoDict['FACE_X'] = facerect[0]
                annoDict['FACE_Y'] = facerect[1]
                annoDict['FACE_WIDTH'] = facerect[2]
                annoDict['FACE_HEIGHT'] = facerect[3]

                row = [annoDict[key] if key in annoDict.keys() else '' for key in SCHEMA]
                print('[vipy.dataset.aflw]: exporting %d points for face "%s" ' % (len(ptsQuery), faceid))
                f.writerow(row)

        db.close()
        return self


def landmarks(im):
    """Return 21x2 frame array of landmark positions in 1-21 order, NaN if occluded"""
    return np.float32(np.array([im.attributes[key] if len(im.attributes[key]) > 0 else np.float32('nan') for key in SCHEMA[5:]])).reshape(21, 2)


def eyes_nose_chin(self, im):
    """Return 4x2 frame array of left eye, right eye nose chin"""
    fr = landmarks(im)
    return fr[[8 - 1, 11 - 1, 15 - 1, 21 - 1],:]  # left eye center, right eye center, nose center  (AFLW annotation, 1-indexed)
