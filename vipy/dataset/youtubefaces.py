import os
from glob import glob
from vipy.util import remkdir, try_import, dirlist, imlist, filetail
from vipy.image import ImageDetection, ImageCategory
try_import('scipy.io', 'scipy')
from scipy.io import loadmat


URL = 'http://www.cslab.openu.ac.il/download/wolftau/YouTubeFaces.tar.gz'


class YouTubeFaces(object):
    def __init__(self, datadir):
        self.datadir = remkdir(datadir)
        self.ytfdir = datadir
        if not os.path.isdir(os.path.join(self.datadir, 'frame_images_DB')):
            raise ValueError('Download YouTubeFaces dataset with "wget %s -O %s; cd %s; tar zxvf YouTubeFaces.tar.gz", and initialize with YouTubeFace(datadir="%s/YouTubeFaces")' % (URL, os.path.join(self.datadir, 'YouTubeFaces.tar.gz'), self.datadir, self.datadir))

    def __repr__(self):
        return str('<viset.youtubefaces: %s>' % self.ytfdir)

    def subjects(self):
        return os.listdir(os.path.join(self.ytfdir, 'descriptors_DB'))

    def videos(self, subject):
        videos = {}
        for d in dirlist(os.path.join(self.ytfdir, 'frame_images_DB', subject)):
            k_videoindex = filetail(d)
            videos[k_videoindex] = []
            for f in imlist(d):
                videos[k_videoindex].append(ImageCategory(filename=f, category=subject))
            videos[k_videoindex] = sorted(videos[k_videoindex], key=lambda im: im.filename())
        return videos

    def parse(self, subject):
        """Parse youtubefaces into a list of ImageDetections"""

        # Write images and annotations
        # The data in this file is in the following format:
        # filename,[ignore],x,y,width,height,[ignore],[ignore]
        # where:
        # x,y are the center of the face and the width and height are of the rectangle that the face is in.
        # For example:
        # $ head -3 Richard_Gere.labeled_faces.txt
        # Richard_Gere\3\3.618.jpg,0,262,165,132,132,0.0,1
        # Richard_Gere\3\3.619.jpg,0,260,164,131,131,0.0,1
        # Richard_Gere\3\3.620.jpg,0,261,165,129,129,0.0,1

        imlist = []
        categorydir = os.path.join(self.ytfdir, 'frame_images_DB')
        for i,infilename in enumerate(glob('%s/*labeled_faces.txt' % categorydir)):
            print('[vipy.dataset.youtubefaces:] parsing "%s" ' % infilename)
            with open(infilename, 'r') as infile:
                for line in infile:
                    (imname, ignore, x_ctr, y_ctr, w, h, ignore, ignore) = line.split(',')
                    imname = imname.replace('\\', '/')
                    category = imname.split('/')[0]
                    xmin = int(x_ctr) - int(float(w) / 2.0)
                    ymin = int(y_ctr) - int(float(h) / 2.0)
                    xmax = int(xmin) + int(w)
                    ymax = int(ymin) + int(h)
                    p = os.path.join(categorydir, imname)

                    (category,videoid,filename) = imname.split('/')
                    mediaID = '%s_%s' % (category, videoid)
                    imlist.append(ImageDetection(filename=p, category=category, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax)).setattribute('MEDIA_ID', str(mediaID)))

        # Done!
        self._parsed = imlist
        return imlist

    def splits(self):
        mat = loadmat(os.path.join(self.ytfdir, 'meta_data', 'meta_and_splits.mat'))
        return mat['Splits']  # splits are indexed into mat['mat_names'] dictionary, parse me!
