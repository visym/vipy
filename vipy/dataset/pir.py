import os
import bobo.app
import numpy as np
from bobo.util import remkdir, isstring, filebase, quietprint, tolist, islist, is_hiddenfile, imlist, filetail, readcsv
from bobo.image import ImageDetection, ImageCategory
from bobo.geometry import BoundingBox
from janus.visualize import montage


class PIR(object):            
    def __init__(self, datadir=None, sparkContext=None):
        self.datadir = os.path.join(bobo.app.datadir(), 'pir_challenge1') if datadir is None else datadir        
        if not os.path.isdir(os.path.join(self.datadir)):
            raise ValueError('Download PIR dataset manually to "%s" ' % self.datadir)
        self.sparkContext = sparkContext

    def __repr__(self):
        return str('<viset.pir: %s>' % self.datadir)
                                                
    def trainset(self, rdd=False):
        visetdir = os.path.join(self.datadir, 'dr')
        imdet = []
        for f in imlist(visetdir):
            f_description = filetail(f).strip().split('_')
            attributes = {'category':f_description[0], 'side':'front' if f_description[6]=='SF' else 'back', 'version':f_description[5], 'extra':f_description[7], 'NationalDrugCode':f_description[0]}
            imdet.append(ImageDetection(filename=os.path.join(self.datadir, 'dr', f), category=f_description[0], bbox=BoundingBox(centroid=(2400/2,1600/2), width=1920, height=1080), attributes=attributes))
        return imdet if rdd==False else self._rdd(imdet)

    def drset(self):
        return self.trainset(rdd=False)
            

    def dcset(self):
        # Ground truth categories
        csvfile = os.path.join(self.datadir, 'pir-products.csv')
        imdet = []
        for pirtruth in readcsv(csvfile):
            f_description = filetail(pirtruth[0][1:-1]).strip().split('_')
            attributes = {'category':f_description[0], 'side':'front' if f_description[6]=='SF' else 'back', 'version':f_description[5], 'extra':f_description[7], 'NationalDrugCode':f_description[0]}                      
            for t in pirtruth[2:]:
                f = t[1:-1]  # strip "quotes" to get filename
                imdet.append(ImageDetection(filename=os.path.join(self.datadir, 'dc', f), category=attributes['category'], attributes=attributes))
        return imdet


    def testset(self, rdd=False):
        # Ground truth categories
        csvfile = os.path.join(self.datadir, 'pir-products.csv')
        imdet = []
        for pirtruth in readcsv(csvfile):
            f_description = filetail(pirtruth[0][1:-1]).strip().split('_')
            attributes = {'category':f_description[0], 'side':'front' if f_description[6]=='SF' else 'back', 'version':f_description[5], 'extra':f_description[7], 'NationalDrugCode':f_description[0]}                      
            for t in pirtruth[2:]:
                f = t[1:-1]  # strip "quotes" to get filename
                imdet.append(ImageDetection(filename=os.path.join(self.datadir, 'dc', f), category=attributes['category'], attributes=attributes))

        # Ground truth bounding boxes (from Brian)                
        imlist = imdet
        dctruth = readcsv(os.path.join(self.datadir, 'dc_med.lst'), separator=' ')  # Brian downsampled by 40% prior to truthing
        dcmap = {filetail(im.filename()):k for (k,im) in enumerate(imlist)}  # mapping from dc filename to index in imlist
        imdet = []
        for t in dctruth:
            if filetail(t[0]) in dcmap.keys():
                im = imlist[dcmap[filetail(t[0])]].clone()
                im = im.boundingbox(xmin=2.5*float(t[1]), ymin=2.5*float(t[2]), width=2.5*float(t[3]), height=2.5*float(t[4]))
                imdet.append(im)
                
        return imdet if rdd==False else self._rdd(imdet)
    

    def split(self, rdd=False, seed=42, tiny=False):
        np.random.seed(int(seed))

        imtrain = self.trainset()
        imtest = self.testset()

        labels = list(set([im.category() for im in imtrain]))
        np.random.shuffle(labels)
        n = int(0.5*len(labels))
        trainlabels = labels[0:n]
        testlabels = labels[n:]

        trainset = [im for im in imtrain if im.category() in trainlabels]
        galleryset = [im for im in imtrain if im.category() in testlabels]        
        trainset = trainset + [im for im in imtest if im.category() in trainlabels]
        probeset = [im for im in imtest if im.category() in testlabels]        
        
        if tiny:
            print '[viset.pir]: using tiny split'
            trainset = list(np.random.choice(trainset, 1000))
            galleryset = list(np.random.choice(galleryset, 100))
            gallerylabels = list(set([im.category() for im in galleryset]))
            probeset = [im for im in probeset if im.category() in gallerylabels]
            probeset = list(np.random.choice(probeset, 100))            
            
        return (trainset, galleryset, probeset) if rdd==False else (self._rdd(trainset), self._rdd(galleryset), self._rdd(probeset))
    
            
    def _rdd(self, imlist):
        """Create a resilient distributed dataset"""
        self.sparkContext = bobo.app.init('viset_pir') if self.sparkContext is None else self.sparkContext
        return self.sparkContext.parallelize(imlist)
        

def testmontage(pir):
    testset = pir.dataset('test')

    imtestset = []    
    for im in testset:
        # Convert to ImageDetection object with central 30% of image 
        imtestset.append(ImageDetection(filename=im.filename(), category=im.category(), bbox=BoundingBox(centroid=(float(im.width())/2.0, float(im.height()/2.0)), width=min(0.3*im.height(), 0.3*im.width()), height=min(0.3*im.height(), 0.3*im.width()))).crop().resize(128,128))
        im.flush() # huge im is loaded and cached on call to im.height()

    return montage(imtestset, 64,64, crop=False, grayscale=False)
    

def trainmontage(pir):
    return montage(pir.dataset('train'), 114, 64, crop=True, grayscale=False)


def stats(pir):
    print len(pir.dataset('train'))
    print len(pir.dataset('test'))

    
def rximage(national_drug_code='00093-7155-98'):
    import requests  # virtualenv
    url = 'http://rximage.nlm.nih.gov/api/rximage/1/rxnav'
    print '[viset.pir.rximage]: querying "%s" ' % url
    r = requests.get(url, params={'ndc':national_drug_code})
    d_rximage = r.json()  # dictionary
    return [ImageCategory(url=d['imageUrl'], attributes=d, category=d['ndc11']) for d in d_rximage['nlmRxImages']]
        

