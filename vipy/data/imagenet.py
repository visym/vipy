import os
import vipy
from vipy.util import readcsv, remkdir, filepath, islist, filetail, filebase, filefull
from vipy.image import ImageDetection, ImageCategory
import xml.etree.ElementTree as ET


IMAGENET_21K_RESIZED_URL = 'https://image-net.org/data/imagenet21k_resized.tar.gz'
URLS_2012 = ['https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz'
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_dogs.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz']
URL_SYNSET = 'https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt'

IMAGENET21K_URL = 'https://image-net.org/data/winter21_wholetar.gz'
IMAGENET21K_WORDNET_ID = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt'
IMAGENET21K_WORDNET_LEMMAS = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'


class Imagenet2012():
    """By downloading, you agree to the ImageNet terms: https://image-net.org/download-images.php#term"""
    def __init__(self, datadir):
        self._datadir = remkdir(datadir)

        for url in URLS_2012:
            if not os.path.exists(os.path.join(datadir, vipy.util.filebase(url))):
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))
                vipy.downloader.unpack(os.path.join(self._datadir, filetail(url)), remkdir(os.path.join(self._datadir, filebase(url))))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train')):
            if not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train_v3')):
            if not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))
                
        if not os.path.exists(os.path.join(self._datadir, 'synset_words.txt')):
            vipy.downloader.download(URL_SYNSET, os.path.join(self._datadir, 'synset_words.txt'))            
        self._synset_to_categorylist = {x.split(' ',1)[0]:[y.lstrip().rstrip() for y in x.split(' ', 1)[1].split(',')] for x in vipy.util.readtxt(os.path.join(self._datadir, 'synset_words.txt'))}            

        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
    
    def classification_trainset(self):
        """ImageNet Classification, trainset"""
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_train'))
        imlist = [vipy.image.ImageCategory(filename=f, category=filetail(filepath(f))) for f in imgfiles]
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification_train')
        
    def classification_valset(self):
        imlist = []
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_val'))

        # ground truth is imagenet synset index 1-1000, which is maped in the metadata
        gt = vipy.util.readtxt(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12', 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'))               
        for (f,y) in zip(sorted(imgfiles), gt):
            imlist.append(vipy.image.ImageCategory(filename=f, category=y))  
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification_val')
                
    def localization_trainset(self):
        """ImageNet localization, imageset = {train, val}"""        
        imlist = []
        classification = self.classification_trainset()
        for im in classification:
            objects = []            
            xmlfile = '%s.xml' % filefull(im.filename().replace('ILSVRC2012_img_train', 'ILSVRC2012_bbox_train_v2'))
            if os.path.exists(xmlfile):
                d = ET.parse(xmlfile).getroot()
                (name, xmin, ymin, xmax, ymax) = (d[5][0].text, d[5][4][0].text, d[5][4][1].text, d[5][4][2].text, d[5][4][3].text)
                objects = [vipy.object.Detection(category=name, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax))]
            imlist.append(vipy.image.Scene(filename=im.filename(), category=im.category(), objects=objects))
                
        return vipy.dataset.Dataset(imlist, 'imagenet2012_localization_train')

    
                
class Imagenet21K_Resized(vipy.dataset.Dataset):
    """https://image-net.org/download-images.php, imagenet-21K 2021 release (resized)"""
    def __init__(self, datadir, aslemma=True):
        self._datadir = vipy.util.remkdir(datadir)
        
        if not os.path.exists(os.path.join(datadir, 'imagenet21k_resized.tar.gz')):
            print('[vipy.data.imagenet]: downloading Imagenet-21K resized to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(IMAGENET_21K_RESIZED_URL, self._outdir, sha1=None)

        if not os.path.exists(os.path.join(self._datadir, 'wordnet_id.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'wordnet_id.txt'))
        if not os.path.exists(os.path.join(self._datadir, 'wordnet_lemmas.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'wordnet_lemmas.txt'))

        # https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_id.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_lemmas.txt')))}

        f_category = lambda c: self._synset_to_categorylist[c][0] if aslemma else c
        imlist = [vipy.image.ImageCategory(filename=f,
                                           attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                           category=f_category(vipy.util.filebase(vipy.util.filepath(f)))) for f in vipy.util.findimages(os.path.join(datadir, 'imagenet21k_resized'))]
        super().__init__(imlist, id='imagenet21k')

    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
        

class Imagenet21K(vipy.dataset.Dataset):
    """https://image-net.org/download-images.php, imagenet-21K 2021 winter release"""
    def __init__(self, datadir, aslemma=True):
        self._datadir = vipy.util.remkdir(datadir)
        
        if not os.path.exists(os.path.join(datadir, 'winter21_whole.tar.gz')):
            print('[vipy.data.imagenet]: downloading Imagenet-21K to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(IMAGENET21K_URL, self._outdir, sha1=None)

        for f in vipy.util.findtar(os.path.join(datadir, 'winter21_whole')):
            if not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))
                os.remove(f)  # cleanup
                
        if not os.path.exists(os.path.join(self._datadir, 'wordnet_id.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'wordnet_id.txt'))
        if not os.path.exists(os.path.join(self._datadir, 'wordnet_lemmas.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'wordnet_lemmas.txt'))

        # https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_id.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_lemmas.txt')))}

        f_category = lambda c: self._synset_to_categorylist[c][0] if aslemma else c
        imlist = [vipy.image.ImageCategory(filename=f,
                                           attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                           category=f_category(vipy.util.filebase(vipy.util.filepath(f)))) for f in vipy.util.findimages(os.path.join(datadir, 'winter21_whole'))]
        super().__init__(imlist, id='imagenet21k')

    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
         
   
