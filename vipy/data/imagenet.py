import os
import vipy
from vipy.util import readcsv, remkdir, filepath, islist, filetail, filebase, filefull, tocache, isinstalled
from vipy.image import ImageDetection, ImageCategory
import xml.etree.ElementTree as ET
import scipy.io
import numpy as np
from vipy.globals import log


URLS_2012 = ['https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',             
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_dogs.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz']

URL_SYNSET = 'https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt'

IMAGENET21K_RESIZED_URL = 'https://image-net.org/data/imagenet21k_resized.tar.gz'
IMAGENET21K_URL = 'https://image-net.org/data/winter21_whole.tar.gz'
IMAGENET21K_WORDNET_ID = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt'
IMAGENET21K_WORDNET_LEMMAS = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'


class Imagenet2012():
    """This requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('imagenet2012') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)

        for url in URLS_2012:
            if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))

        for url in URLS_2012:
            if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
                vipy.downloader.unpack(os.path.join(self._datadir, filetail(url)), remkdir(os.path.join(self._datadir, filebase(url))))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train')):
            if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
                vipy.downloader.unpack(f, filefull(f))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train_v3')):
            if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
                vipy.downloader.unpack(f, filefull(f))
                
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download(URL_SYNSET, os.path.join(self._datadir, 'synset_words.txt'))            
        self._synset_to_categorylist = {x.split(' ',1)[0]:[y.lstrip().rstrip() for y in x.split(' ', 1)[1].split(',')] for x in vipy.util.readtxt(os.path.join(self._datadir, 'synset_words.txt'))}            

        metadata = scipy.io.loadmat(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'), struct_as_record=False)
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
        wnids = np.squeeze(np.array([s.WNID for s in synsets]))
        words = np.squeeze(np.array([s.words for s in synsets]))

        self._wnid_to_categorylist = {wnid:[c.strip() for c in category.split(',')] for (wnid,category) in zip(wnids, words)}
                                    
        open(os.path.join(self._datadir, '.complete'), 'a').close()        
        
    def synset_to_category(self, s=None):
        return self._wnid_to_categorylist if s is None else self._wnid_to_categorylist[s]
    
    def classification_trainset(self):
        """ImageNet Classification, trainset"""
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_train'))
        loader = lambda f, synset_to_category=self.synset_to_category: vipy.image.TaggedImage(filename=f, tags=synset_to_category(filetail(filepath(f))))
        return vipy.dataset.Dataset(imgfiles, 'imagenet2012_classification:train', loader=loader)
        
    def classification_valset(self):
        imlist = []
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_val'))
                    
        # ground truth is imagenet synset index 1-1000
        gt = vipy.util.readtxt(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12', 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'))
        for (f,y) in zip(sorted(imgfiles), gt):
            imlist.append( (f, y) )

        # Index mapping is in mat file (yuck)
        synsets = self.synset_to_category()
        d_idx_to_category = {str(k):self.synset_to_category(r[0][1][0]) for (k,r) in enumerate(scipy.io.loadmat(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'))['synsets'], start=1) if r[0][1][0] in synsets}
        loader = lambda x, d_idx_to_category=d_idx_to_category: vipy.image.TaggedImage(filename=x[0], tags=d_idx_to_category[x[1]])
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification:val', loader=loader)
                
    def localization_trainset(self):
        """ImageNet localization, imageset = {train, val}, this takes a long time to read the XML files, load and cache"""
        log.warning('Parsing XML files for imagenet-localization takes a long time...')
        
        imlist = []
        classification = self.classification_trainset()
        synsets = self.synset_to_category()
        for f in classification._ds:
            
            objects = []            
            xmlfile = '%s.xml' % filefull(f.replace('ILSVRC2012_img_train', 'ILSVRC2012_bbox_train_v2'))
            if os.path.exists(xmlfile):
                d = ET.parse(xmlfile).getroot()
                (name, xmin, ymin, xmax, ymax) = (d[5][0].text, d[5][4][0].text, d[5][4][1].text, d[5][4][2].text, d[5][4][3].text)
                objects.append( (synsets[name] if name in synsets else name, xmin, ymin, xmax, ymax) )
            imlist.append( (f,tuple(objects)) )

        loader = lambda x, synset_to_category=self.synset_to_category: vipy.image.Scene(filename=x[0],
                                                                                        category=synset_to_category(filetail(filepath(x[0]))),
                                                                                        objects=[vipy.object.Detection(category=o[0], xmin=int(o[1]), ymin=int(o[2]), xmax=int(o[3]), ymax=int(o[4])) for o in x[1]])
        return vipy.dataset.Dataset(imlist, 'imagenet2012_localization:train', loader=loader)

    
                
class Imagenet21K_Resized(vipy.dataset.Dataset):
    """https://image-net.org/download-images.php, imagenet-21K 2021 release ("squish" resized)"""
    def __init__(self, datadir=None, aslemma=True, redownload=False, recache=False):

        datadir = tocache('imagenet21k_resized') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)
        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.globals.log.info('[vipy.data.imagenet]: downloading Imagenet-21K resized to "%s"' % self._datadir)            
            vipy.downloader.download_and_unpack(IMAGENET21K_RESIZED_URL, self._datadir, sha1=None)

        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt'))
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt'))

        # Class names: https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt')))}

        cachefile = os.path.join(self._datadir, '.imlist.txt')
        if recache and os.path.exists(cachefile):
            os.remove(cachefile)
            
        f_category = lambda c, synset_to_categorylist=self._synset_to_categorylist, aslemma=aslemma: synset_to_categorylist[c][0] if aslemma else c        
        imlist = vipy.util.findimages(os.path.join(datadir, 'imagenet21k_resized')) if not os.path.exists(cachefile) else [os.path.join(self._datadir, f) for f in vipy.util.readlist(cachefile)]
        loader = lambda f, f_category=f_category: vipy.image.TaggegdImage(filename=f,
                                                                          attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                                                          tags=f_category(vipy.util.filebase(vipy.util.filepath(f))))
        super().__init__(imlist, id='imagenet21k_resized', loader=loader)

        if not os.path.exists(cachefile):
            vipy.util.writelist([f.replace(self._datadir + '/', '') for f in imlist], cachefile)  # cache me for faster loading instead of walking the directory tree        
        open(os.path.join(self._datadir, '.complete'), 'a').close()

        
        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
        

class Imagenet21K(vipy.dataset.Dataset):
    """This requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term    
       imagenet-21K 2021 winter release
    """
    def __init__(self, datadir=None, aslemma=True, redownload=False, recache=False):

        datadir = tocache('imagenet21k') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)
        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            if isinstalled('wget'):
                vipy.globals.log.info('downloading "%s" to "%s"' % (IMAGENET21K_URL , self._datadir))                
                os.system('wget --no-check-certificate --continue --tries=32 -O %s %s ' % (os.path.join(self._datadir, filetail(IMAGENET21K_URL)), IMAGENET21K_URL))  # server fails many times, need smart continue
            else:
                vipy.downloader.download(IMAGENET21K_URL, os.path.join(self._datadir, filetail(IMAGENET21K_URL)))  # fallback on dumb downloader
            vipy.downloader.unpack(os.path.join(self._datadir, filetail(IMAGENET21K_URL)), self._datadir)  # fallback on dumb downloader

            for f in vipy.util.findtar(os.path.join(datadir, 'winter21_whole')):
                if not os.path.exists(filefull(f)):
                    vipy.downloader.unpack(f, filefull(f))
                    os.remove(f)  # cleanup
                
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt'))
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt'))

        cachefile = os.path.join(self._datadir, '.imlist.txt')
        if recache and os.path.exists(cachefile):
            os.remove(cachefile)
            
        # Class names: https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:sorted([y.rstrip().lstrip() for y in lemma.split(',')]) for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt')))}

        f_category = lambda c, synset_to_categorylist=self._synset_to_categorylist, aslemma=aslemma: synset_to_categorylist[c] if aslemma else c
        imlist = vipy.util.findimages(os.path.join(datadir, 'winter21_whole')) if not os.path.exists(cachefile) else [os.path.join(self._datadir, f) for f in vipy.util.readlist(cachefile)]
        loader = lambda f, f_category=f_category: vipy.image.TaggedImage(filename=f,
                                                                         attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                                                         tags=f_category(vipy.util.filebase(vipy.util.filepath(f))))
        super().__init__(imlist, id='imagenet21k', loader=loader)

        if not os.path.exists(cachefile):
            vipy.util.writelist([f.replace(self._datadir + '/', '') for f in imlist], cachefile)  # cache me for faster loading instead of walking the directory tree

        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
         
   
