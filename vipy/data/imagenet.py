import os
import vipy
from vipy.util import readcsv, remkdir, filepath, islist, filetail, filebase, filefull, tocache
from vipy.image import ImageDetection, ImageCategory
import xml.etree.ElementTree as ET
import scipy.io


URLS_2012 = ['https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',             
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_dogs.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz']

URL_SYNSET = 'https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt'

IMAGENET21K_RESIZED_URL = 'https://image-net.org/data/imagenet21k_resized.tar.gz'
IMAGENET21K_URL = 'https://image-net.org/data/winter21_wholetar.gz'
IMAGENET21K_WORDNET_ID = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt'
IMAGENET21K_WORDNET_LEMMAS = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'


class Imagenet2012():
    """This requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term"""
    def __init__(self, datadir=tocache('imagenet2012'), redownload=False):
        self._datadir = remkdir(datadir)

        for url in URLS_2012:
            if redownload or not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))

        for url in URLS_2012:
            if redownload or not os.path.exists(os.path.join(datadir, vipy.util.filebase(url))):            
                vipy.downloader.unpack(os.path.join(self._datadir, filetail(url)), remkdir(os.path.join(self._datadir, filebase(url))))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train')):
            if redownload or not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))

        for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train_v3')):
            if redownload or not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))
                
        if redownload or not os.path.exists(os.path.join(self._datadir, 'synset_words.txt')):
            vipy.downloader.download(URL_SYNSET, os.path.join(self._datadir, 'synset_words.txt'))            
        self._synset_to_categorylist = {x.split(' ',1)[0]:[y.lstrip().rstrip() for y in x.split(' ', 1)[1].split(',')] for x in vipy.util.readtxt(os.path.join(self._datadir, 'synset_words.txt'))}            

        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
    
    def classification_trainset(self):
        """ImageNet Classification, trainset"""
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_train'))
        loader = lambda f, synset_to_category=self.synset_to_category: vipy.image.ImageCategory(filename=f, category=','.join(synset_to_category(filetail(filepath(f)))))
        return vipy.dataset.Dataset(imgfiles, 'imagenet2012_classification_train', loader=loader)
        
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
        loader = lambda x, d_idx_to_category=d_idx_to_category: vipy.image.ImageCategory(filename=x[0], category=','.join(d_idx_to_category[x[1]]))
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification_val', loader=loader)
                
    def localization_trainset(self):
        """ImageNet localization, imageset = {train, val}, this takes a long time to read the XML files, load and cache"""        
        imlist = []
        classification = self.classification_trainset()
        synsets = self.synset_to_category()
        for f in classification._ds:
            
            objects = []            
            xmlfile = '%s.xml' % filefull(f.replace('ILSVRC2012_img_train', 'ILSVRC2012_bbox_train_v2'))
            if os.path.exists(xmlfile):
                d = ET.parse(xmlfile).getroot()
                (name, xmin, ymin, xmax, ymax) = (d[5][0].text, d[5][4][0].text, d[5][4][1].text, d[5][4][2].text, d[5][4][3].text)
                objects.append( (','.join(synsets[name]) if name in synsets else name, xmin, ymin, xmax, ymax) )
            imlist.append( (f,tuple(objects)) )

        loader = lambda x, synset_to_category=self.synset_to_category: vipy.image.Scene(filename=x[0],
                                                                                        category=','.join(synset_to_category(filetail(filepath(x[0])))),
                                                                                        objects=[vipy.object.Detection(category=o[0], xmin=int(o[1]), ymin=int(o[2]), xmax=int(o[3]), ymax=int(o[4])) for o in x[1]])
        return vipy.dataset.Dataset(imlist, 'imagenet2012_localization_train', loader=loader)

    
                
class Imagenet21K_Resized(vipy.dataset.Dataset):
    """https://image-net.org/download-images.php, imagenet-21K 2021 release (resized)"""
    def __init__(self, datadir=tocache('imagenet21k_resized'), aslemma=True, redownload=False):
        self._datadir = vipy.util.remkdir(datadir)
        
        if redownload or not os.path.exists(os.path.join(datadir, 'imagenet21k_resized.tar.gz')):
            print('[vipy.data.imagenet]: downloading Imagenet-21K resized to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(IMAGENET21K_RESIZED_URL, self._outdir, sha1=None)

        if redownload or not os.path.exists(os.path.join(self._datadir, 'wordnet_id.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'wordnet_id.txt'))
        if redownload or not os.path.exists(os.path.join(self._datadir, 'wordnet_lemmas.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'wordnet_lemmas.txt'))

        # https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_id.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_lemmas.txt')))}

        f_category = lambda c, synset_to_categorylist=self._synset_to_categorylist, aslemma=aslemma: synset_to_categorylist[c][0] if aslemma else c
        imlist = vipy.util.findimages(os.path.join(datadir, 'imagenet21k_resized'))
        loader = lambda f, f_category=f_category: vipy.image.ImageCategory(filename=f,
                                                                           attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                                                           category=f_category(vipy.util.filebase(vipy.util.filepath(f))))
        super().__init__(imlist, id='imagenet21k_resized', loader=loader)

    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
        

class Imagenet21K(vipy.dataset.Dataset):
    """https://image-net.org/download-images.php, imagenet-21K 2021 winter release"""
    def __init__(self, datadir=tocache('imagenet21k'), aslemma=True, redownload=False):
        self._datadir = vipy.util.remkdir(datadir)
        
        if redownload or not os.path.exists(os.path.join(datadir, 'winter21_whole.tar.gz')):
            print('[vipy.data.imagenet]: downloading Imagenet-21K to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(IMAGENET21K_URL, self._outdir, sha1=None)

        for f in vipy.util.findtar(os.path.join(datadir, 'winter21_whole')):
            if not os.path.exists(filefull(f)):
                vipy.downloader.unpack(f, filefull(f))
                os.remove(f)  # cleanup
                
        if redownload or not os.path.exists(os.path.join(self._datadir, 'wordnet_id.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'wordnet_id.txt'))
        if redownload or not os.path.exists(os.path.join(self._datadir, 'wordnet_lemmas.txt')):
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'wordnet_lemmas.txt'))

        # https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_id.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'wordnet_lemmas.txt')))}

        f_category = lambda c, synset_to_categorylist=self._synset_to_categorylist, aslemma=aslemma: synset_to_categorylist[c][0] if aslemma else c
        imlist = vipy.util.findimages(os.path.join(datadir, 'winter21_whole'))
        loader = lambda f, f_category=f_category: vipy.image.ImageCategory(filename=f,
                                                                           attributes={'wordnet_id':vipy.util.filebase(vipy.util.filepath(f))},
                                                                           category=f_category(vipy.util.filebase(vipy.util.filepath(f))))
        super().__init__(imlist, id='imagenet21k', loader=loader)
        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
         
   
