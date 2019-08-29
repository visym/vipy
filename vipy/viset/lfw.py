import os
from bobo.util import remkdir, isstring, quietprint, filetail, dirlist, imlist
from bobo.image import ImageDetection
from bobo.geometry import BoundingBox
import bobo.app
import numpy as np


URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
URL_DEEPFUNNEL = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
URL_FUNNEL = 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz'
URL_LFWA = 'http://www.openu.ac.il/home/hassner/data/lfwa/lfwa.tar.gz'
URL_NAMES = 'http://vis-www.cs.umass.edu/lfw/lfw-names.txt'
URL_PAIRS_DEV_TRAIN = 'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt'
URL_PAIRS_DEV_TEST = 'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt'
URL_PAIRS_VIEW2 = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'

class LFW(object):
    def __init__(self, datadir=bobo.app.datadir(), cache_root=bobo.app.janusCache(), funneled=None, bbox_size=(125,150)):
        self.datadir = datadir
        if funneled == None:
            self.lfwdir = os.path.join(self.datadir, 'lfw')
            self.lfwpairsdir = self.lfwdir
        elif funneled == 'funneled':
            self.lfwdir = os.path.join(self.datadir, 'lfw_funneled', 'lfw_funneled')
            self.lfwpairsdir = os.path.join(self.datadir, 'lfw_funneled')
        elif funneled == 'deepfunneled':
            self.lfwdir = os.path.join(self.datadir, 'lfw_deepfunneled', 'lfw-deepfunneled')
            self.lfwpairsdir = os.path.join(self.datadir, 'lfw_deepfunneled')
        self.chipdir_base = os.path.join(cache_root, "pvr", "chips")
        self.frontdir_base = os.path.join(cache_root, "pvr", "frontalized", "frontalized")
        self.frontdir_base = os.path.join(cache_root, "pvr", "frontalized", "frontalized_raw")
        self.metadir_base = os.path.join(cache_root, "pvr", "frontalized", "meta")
        
        if not os.path.isdir(os.path.join(self.lfwdir, 'AJ_Cook')):
            raise ValueError('Download LFW dataset manually and unpack to to "%s" ' % self.lfwdir)
        if not os.path.isfile(os.path.join(self.lfwpairsdir, 'pairsDevTrain.txt')):
            raise ValueError('Download LFW pairsDevTrain.txt manually and save in "%s"" ' % self.lfwpairsdir)
        if not os.path.isfile(os.path.join(self.lfwpairsdir, 'pairsDevTest.txt')):
            raise ValueError('Download LFW pairsDevTest.txt manually and save in "%s"" ' % self.lfwpairsdir)
        if not os.path.isfile(os.path.join(self.lfwpairsdir, 'pairs.txt')):
            raise ValueError('Download LFW pairs.txt manually and save in "%s"" ' % self.lfwpairsdir)
        
        if bbox_size is None:
            self.f_bbox = lambda im: None
        elif bbox_size == "viola_jones":
            raise Exception("Not implemented")
        elif bbox_size == "dlib":
            raise Exception("Not implemented")
        else:
            try:
                width,height = bbox_size
            except:
                width, height = map(int, bbox_size.split('x'))
            self.f_bbox = lambda im: BoundingBox(xmin=(im.width()/2-width/2), ymin=(im.height()/2-height/2),
                                                    width=width, height=height)
            self.bbox = BoundingBox(centroid=(125,125), width=width, height=height)

    def __repr__(self):
        return str('<viset.lfw: %s>' % self.lfwdir)

    def subjects(self):
        """List of all subject names"""
        return [filetail(d) for d in dirlist(self.lfwdir) if filetail(d) != 'lfw']

    def _subject_images(self, subject):
        """List of Images of a subject"""
        fnames = imlist(os.path.join(self.lfwdir,subject))
        ks = [fname[-8:-4] for fname in fnames]     # filename is '.../<category>_<4-digit k>.jpg', so extract just the 4-digit k's
        return [self._ImageDetectionFromTuple(self._makeParsedTuple(subject, k)) for k in ks]

    def view_subjects(self):
        """List of all Images of all subjects"""
        quietprint('[viset.lfw][LFW.view_subjects] Traversing %s recursively for images'%self.lfwdir, verbosity=2)
        return [self._subject_images(subject) for subject in self.subjects()]
    
    def _makeParsedTuple(self, category, k):
        sighting_id = '%s_%04d'%(category, int(k))
        return category, os.path.join(self.lfwdir, category, sighting_id+'.jpg'), sighting_id

    def _ImageDetectionFromTuple(self, p):
        im = ImageDetection(filename=p[1], category=p[0], bbox=self.bbox.clone(), attributes={'SIGHTING_ID': p[2], 'TEMPLATE_ID': p[0]})
        #im.boundingbox(bbox=self.f_bbox(im))
        return im

    def _parseLine(self, line):
        row = line.split()
        if len(row) == 3:
            category, k_probe, k_gallery = row
            return self._makeParsedTuple(category, k_probe), \
                   self._makeParsedTuple(category, k_gallery)
        elif len(row) == 4:
            lbl_probe, k_probe, lbl_gallery, k_gallery = row
            return self._makeParsedTuple(lbl_probe,   k_probe), \
                   self._makeParsedTuple(lbl_gallery, k_gallery)
        else: raise ValueError(line)

    def _parse_view1_protocol2(self):
        """ View 1 (development training/testing split) - Protocol 2 (image restricted verification) """
        infiles = [os.path.join(self.lfwpairsdir, 'pairsDevTrain.txt'), os.path.join(self.lfwpairsdir, 'pairsDevTest.txt')]
        outlists = [[], []]
        for (infile, outlist) in zip(infiles, outlists):
            with open(infile, 'r') as f_read:
                expected_len = int(f_read.readline())
                outlist.extend( self._parseLine(line) for line in f_read )
                assert len(outlist) == expected_len*2
        (trainlist, testlist) = outlists;
        return (trainlist, testlist)

    def _parse_view2_protocol2(self):                                    
        """ View 2 (reporting 10-fold cross validation) - Protocol 2 (image restricted verification)"""
        infile = os.path.join(self.lfwpairsdir, 'pairs.txt')
        with open(infile, 'r') as f_read:
            expected_splits, expected_split_lens = map(int, f_read.readline().split())
            splits = [[self._parseLine(f_read.readline()) for _ in xrange(expected_split_lens*2)] for _ in xrange(expected_splits)]

            # check that parsing went well
            assert len(f_read.readlines()) == 0 # no extra lines
            assert len(splits) == expected_splits
            for s in splits: assert len(s) == expected_split_lens*2
        # Done!
        return (splits)

    def view1_protocol2(self):
        """Face verification development splits (view1), image restricted protocol with no outside data (protocol 2), See also:  http://vis-www.cs.umass.edu/lfw/lfw_update.pdf"""        
        (trainlist, testlist) = self._parse_view1_protocol2()
        imtestlist = [(self._ImageDetectionFromTuple(p),
                       self._ImageDetectionFromTuple(q)) for (p,q) in testlist]
        imtrainlist = [(self._ImageDetectionFromTuple(p),
                        self._ImageDetectionFromTuple(q)) for (p,q) in trainlist]
        return [imtrainlist, imtestlist]

    def view1_protocol2_train(self):
        """Face verification development splits (view1), image restricted protocol with no outside data (protocol 2), See also:  http://vis-www.cs.umass.edu/lfw/lfw_update.pdf"""        
        trainlist,_ = self._parse_view1_protocol2()
        imtrainlist = [(self._ImageDetectionFromTuple(p),
                        self._ImageDetectionFromTuple(q)) for (p,q) in trainlist]
        return (imtrainlist,)
    
    def view2_protocol2(self):
        """Face verification 10-fold training/testing splits (view 2), image restricted protocol with no outside data (protocol 2), See also:  http://vis-www.cs.umass.edu/lfw/lfw_update.pdf"""        
        splitlist = self._parse_view2_protocol2()
        return [[(self._ImageDetectionFromTuple(p),
                  self._ImageDetectionFromTuple(q)) for (p,q) in s] for s in splitlist]

    def get_frontalized_dir(self, im):
        raise Exception("LFW.get_frontalized_dir() shouldn't be used")
        frontdir = os.path.join(self.frontdir_base, im.category())
        #bobo.util.remkdir(frontdir)
        return frontdir

    def get_frontalized_raw_dir(self, im):
        raise Exception("LFW.get_frontalized_raw_dir() shouldn't be used")
        frontrawdir = os.path.join(self.frontrawdir_base, im.category())
        #bobo.util.remkdir(frontrawdir)
        return frontrawdir

    def get_meta_dir(self, im):
        raise Exception("LFW.get_meta_dir() shouldn't be used")
        metadir = os.path.join(self.metadir_base, im.category())
        #bobo.util.remkdir(metadir)
        return metadir

    def get_chip_path(self, im):
        """
        Get the path of where the chip will be saved.

        Also makes sure the directry exists.

        Assumes im is from the LFW dataset, of course.
        """
        raise Exception("LFW.get_chip_pah() shouldn't be used")
        chipdir = os.path.join(self.chipdir_base, im.category())
        #bobo.util.remkdir(chipdir)
        return os.path.join(chipdir, '%s.png'%im.attributes['SIGHTING_ID'])

    def view(self, view_str):
        if view_str == 'subjects':
            views = self.view_subjects()
        elif view_str == 'view1_protocol2':
            views = self.view1_protocol2()
        elif view_str == 'view1_protocol2_train':
            views = self.view1_protocol2_train()
        elif view_str == 'view2_protocol2':
            views = self.view2_protocol2()
        elif view_str == 'view2_protocol2_debug':
            views = self.view2_protocol2()
            del views[:8]
            del views[0][:551]
        else:
            raise ValueError('Unknown view "%s"'%args.view)
        return views

    def flatview_pairs(self, split):
        def sid(im): return im.attributes['SIGHTING_ID']
        def cat(im): return im.category()
        id_inds = {}
        imgs = []
        pairs = []
        def ind(im):
            try: return id_inds[sid(im)]
            except:
                ind = len(imgs)
                id_inds[sid(im)] = ind
                imgs.append(im)
                return ind
        for (a,b) in split:
            aind = ind(a)
            bind = ind(b)
            pairs.append((aind, bind, a.category() == b.category()))
        return imgs, pairs

    def sample(self, n=128):
        """Return a represenative list of 128 images"""        
        imlist = [p[0] for p in self.view1_protocol2()[0]]  # test set from view 1 for a list of images
        return np.random.choice(imlist, n)
    
