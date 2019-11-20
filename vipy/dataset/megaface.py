import os
from bobo.util import remkdir, dirlist, imlist, filebase, readcsv, writecsv
from bobo.image import ImageDetection
import bobo.app
import json
import numpy as np



class MF2(object):
    def __init__(self, datadir='/proj/janus3/megaface'):
        self.datadir = datadir

    def __repr__(self):
        return str('<viset.megaface: %s>' % self.datadir)

    def _trainset(self):
        """Save a csv file containing each image on a line for Megaface_Challenge_1M_disjoint_LOOSE.tar.gz"""
        outfile = os.path.join(self.datadir, 'Megaface_Challenge_1M_disjoint_LOOSE.csv')
        subdir = os.path.join(self.datadir, 'Megaface_Challenge_1M_disjoint_LOOSE')
        D =  dirlist(subdir)
        filelist = []
        for (k,d) in enumerate(D):
            print '[MF2.trainset][%d/%d]: creating image list for "%s"' % (k, len(D), d)            
            for f in imlist(d):
                filelist.append((f, filebase(d)))
        return writecsv(filelist, outfile);

    def tinyset(self, size=1000):
        """Return the first (size) image objects in the trainset"""
        outlist = []
        if not os.path.exists(os.path.join(self.datadir, 'Megaface_Challenge_1M_disjoint_LOOSE.csv')):
            print '[MF2.tinyset]: generating csv file for MF2'
            self._trainset()

        imglist = np.random.permutation([f[0] for f in readcsv(os.path.join(self.datadir, 'Megaface_Challenge_1M_disjoint_LOOSE.csv'))])
        for (k,f) in enumerate(imglist):
            print '[MF2.tinyset][%d/%d]: importing "%s"' % (k, size, f)            
            outlist = outlist + [ImageDetection(filename=os.path.join(self.datadir, f), category=filebase(f))]
            if k > size:
                break 
        return outlist
        

class Megaface(object):
    def __init__(self, datadir=None):
        self.datadir = bobo.app.datadir() if datadir is None else datadir

    def __repr__(self):
        return str('<viset.megaface: %s>' % self.datadir)

    def _attributes(self, imgfile):
        return json.load(open(imgfile + '.json', 'r'))

    def _imagelist(self):
        """Save a csv file containing each image on a line"""
        if os.path.exists(os.path.join(self.datadir, 'megaface.csv')):
            return(os.path.join(self.datadir, 'megaface.csv'))
        else:
            outfile = os.path.join(self.datadir, 'megaface.csv')
            with open(outfile, 'w') as csv:
                subdir = os.path.join(self.datadir, 'FlickrFinal2')
                D =  dirlist(subdir)
                for (k,d) in enumerate(D):
                    print '[megaface.dataset][%d/%d]: creating image list for "%s"' % (k, len(D), d)            
                    for sd in dirlist(d):
                        for f in imlist(sd):
                            csv.write(f + '\n')  # full path
            return outfile

    def tinyset(self, size=1000):
        """Return the first (size) image objects in the dataset"""
        outlist = []
        imglist = np.random.permutation([f[0] for f in readcsv(self._imagelist())])
        for (k,f) in enumerate(imglist):
            print '[megaface.dataset][%d/%d]: importing "%s"' % (k, size, f)            
            A = self._attributes(os.path.join(self.datadir, f))
            outlist = outlist + [ImageDetection(filename=os.path.join(self.datadir, f), category=filebase(f)).boundingbox(xmin=A['bounding_box']['x'], ymin=A['bounding_box']['y'], width=A['bounding_box']['width'], height=A['bounding_box']['height'])]
            if k > size:
                break 
        return outlist
    
    def rdd(self, appname='megaface'):
        return (bobo.app.init(appname).textFile(self._imagelist())
                    .map(lambda f: ImageDetection(filename=os.path.join(self.datadir, f), category=filebase(f), attributes=self._attributes(os.path.join(self.datadir, f))))
                    .map(lambda im: im.boundingbox(xmin=im.attributes['bounding_box']['x'], ymin=im.attributes['bounding_box']['y'], width=im.attributes['bounding_box']['width'], height=im.attributes['bounding_box']['height'])))
                                
