#!/usr/bin/env python
"""FDDB dataset module"""

from __future__ import print_function
import os
import math
from collections import defaultdict
import bobo.app
from bobo.util import quietprint, remkdir, Stopwatch, saveas, filetail, filepath, filebase, islist
from bobo.image import Image, ImageDetection
from bobo.geometry import BoundingBox, Ellipse
from bobo.cache import Cache

from janus.detection import  DlibObjectDetector

class FDDB(object):
    """Manages the FDDB dataset: http://vis-www.cs.umass.edu/fddb"""

    def __init__(self, rootdir=None):
        self.root_dir = (rootdir if rootdir is not None
                         else os.path.join(bobo.app.datadir(), 'fddb'))
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.folds_dir = os.path.join(self.root_dir, 'FDDB-folds')

    def cachecheck(self):
        if not os.path.isdir(os.path.join(self.img_dir, '2002')):
            """ The tarball contains multiple copies of some images with read-only permissions.
            This causes the tarfile module to throw an IOError partway through extraction when
            trying to overwite.  Ask the user to download and extract manually.  """

            data_url = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
            raise ValueError('Download FDDB dataset manually and unpack to to "%s" ' % self.img_dir)

        if not os.path.isfile(os.path.join(self.folds_dir, 'FDDB-fold-01.txt')):
            remkdir(self.root_dir)
            folds_url = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
            folds_sha1 = '94ce19ba3348425dfbb8bcbe55802490f8f152f8'
            folds_file = os.path.join(self.root_dir, filetail(folds_url))
            cache = Cache()
            cache.get(url=folds_url, sha1=folds_sha1, key=folds_file)
            cache.unpack(folds_file, unpackto=self.root_dir)
        return True


    def imset(self, fold=1):
        if not self.cachecheck():
            return
        fold_file = os.path.join(self.folds_dir, 'FDDB-fold-%02d.txt' % fold)
        imset = []
        with open(fold_file, 'r') as f:
            for line in f:
                filename = self.unkey(line.strip())
                imset.append(Image(filename=filename))
        return imset


    def groundtruth(self, fold=1):
        if not self.cachecheck():
            return
        fold_file = os.path.join(self.folds_dir,
                                 'FDDB-fold-%02d-ellipseList.txt' % fold)
        ground_truth = defaultdict(list)
        with open(fold_file, 'r') as f:
            for line in f:
                # image name
                filename = self.unkey(line.strip())
                dets = []
                # next line has truth count
                detcnt = next(f)
                for i in range(int(detcnt)):
                    # Detection
                    det = next(f)
                    (maj_rad, min_rad, angle, center_x, center_y,
                     score) = [float(z) for z in det.split()]
                    # FIXME: Assumes semi-major is vertical
                    xmin = center_x - min_rad
                    ymin = center_y - maj_rad
                    xmax = center_x + min_rad
                    ymax = center_y + maj_rad
                    ellipse = Ellipse(semi_major=maj_rad, semi_minor=min_rad, phi=angle,
                                    xcenter=center_x, ycenter=center_y)
                    bbox = ellipse.estimate_boundingbox()
                    bbox.score = 1.0
                    d = dict(ellipse=ellipse, bbox=bbox)
                    ground_truth[filename].append(d)
        return dict(ground_truth)


    def folds(self, folds=1):
        if not islist(folds):
            folds = [folds]
        truths = []
        imsets = []
        for fold in folds:
            truths.append(self.groundtruth(fold=fold))
            imsets.append(self.imset(fold=fold))
        return (imsets, truths)


    def detectionset(self, fold=1, name=None):
        imset = self.imagedetectionset(fold=fold, name=name)
        dets = defaultdict(list)

        for im in imset:
            fn = im.filename()
            dets[fn].append(im.boundingbox())
        return dets


    def imagedetectionset(self, fold=1, name=None):
        dets_dir = os.path.join(self.root_dir, name)
        fold_file = os.path.join(dets_dir, 'fold-%02d-out.txt' % fold)
        imset = []
        with open(fold_file, 'r') as f:
            for line in f: # We will advance this iterator inside the loop as well
                filename = self.unkey(line.strip())
                detcnt = next(f)
                for i in range(int(detcnt)):
                    det = next(f)
                    vals = det.split()
                    if len(vals) == 5:
                        (xmin, ymin, width, height, score) = [float(z) for z in vals]
                        xmax = xmin + width
                        ymax = ymin + height
                    else:
                        (smaj, smin, angle, center_x, center_y, score) = [float(z) for z in vals]
                        xmin = center_x - smin
                        ymin = center_y - smaj
                        xmax = center_x + smin
                        ymax = center_y + smaj
                    bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score=score)
                    imset.append(ImageDetection(filename=filename, bbox=bbox))
        return imset


    def key(self, filename):
        """FDDB_DIR/img/2002/08/11/big/img_591.jpg => 2002/08/11/big/img_591"""
        return '/'.join(filename.split('/')[-5:])[:-4]


    def unkey(self, key):
        """2002/08/11/big/img_591 => FDDB_DIR/img/2002/08/11/big/img_591.jpg"""
        imgname = key + '.jpg'
        filename = os.path.join(self.img_dir, imgname)
        return filename


    def writedetections(self, name, fold, detections,
                        write_images=False, do_ellipse=False):
        """ Takes a list of tuples where the first item is image key, and the
            second is a list of bounding box detections.  Writes output in FDDB
            detectionformat """

        outfile = os.path.join(self.root_dir, name, 'fold-%02d-out.txt' % fold)
        remkdir(bobo.util.filepath(outfile))
        with open(outfile, 'w') as f:
            for k, v in detections:
                key = self.key(k)
                print(key, file=f)
                print(str(len(v)), file=f)
                for d in v:
                    if do_ellipse:
                        """Bounding box format: semimajor_radius,
                        semiminor_radius, angle, center_x, center_y, score"""
                        cx, cy = d.centroid()
                        h = d.height()
                        w = d.width()
                        a = math.pi / 2 # semimajor is vertical
                        print('{0} {1} {2} {3} {4} {5}'.format(h/2, w/2, a, cx, cy, d.score), file=f)
                    else:
                        # Bounding box format: left, top, width, height, score
                        print('{0} {1} {2} {3} {4}'.format(d.xmin, d.ymin, d.width(), d.height(), d.score), file=f)
                if write_images:
                    image_dir = os.path.join(self.root_dir, name, filepath(key))
                    remkdir(image_dir)
                    im = Image(filename=k)
                    im.load()
                    for d in v:
                        im.drawbox(bbox=d)
                    fn = filebase(im.filename() + '.png')
                    out_img = os.path.join(image_dir, fn)
                    quietprint('Saving %s' % out_img, 2)
                    im.saveas(out_img)
