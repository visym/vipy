import os
import vipy
from vipy.util import readcsv, remkdir, filepath, islist, filetail, filebase, filefull, tocache, isinstalled
from vipy.image import ImageDetection, ImageCategory
import numpy as np
from vipy.globals import log

try:
    import lxml.etree as ET  # faster (optional)
except:
    import xml.etree.ElementTree as ET  # slower (default)

    
URLS_2014 = ['https://image-net.org/data/ILSVRC/2014/ILSVRC2014_DET_train.tar',
             'https://image-net.org/data/ILSVRC/2013/ILSVRC2013_DET_val.tar',
             'https://image-net.org/data/ILSVRC/2014/ILSVRC2014_DET_bbox_train.tgz',
             'https://image-net.org/data/ILSVRC/2013/ILSVRC2013_DET_bbox_val.tgz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz',
             'https://image-net.org/data/ILSVRC/2013/ILSVRC2013_DET_test.tar',
             'https://image-net.org/data/bboxes_annotations.tar.gz']
             
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
IMAGENET21K_MD5 = 'ab313ce03179fd803a401b02c651c0a2'

IMAGENET_FACES = 'https://image-net.org/data/face_annotations_ILSVRC.json'


class Imagenet2012():
    """Imagenet2012 requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('imagenet2012') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)

        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):        
            for url in URLS_2012:
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))

            for url in URLS_2012:
                vipy.downloader.unpack(os.path.join(self._datadir, filetail(url)), remkdir(os.path.join(self._datadir, filebase(url))))

            for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train')):
                vipy.downloader.unpack(f, filefull(f))

            for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2012_img_train_v3')):
                vipy.downloader.unpack(f, filefull(f))
               
            vipy.downloader.download(URL_SYNSET, os.path.join(self._datadir, 'synset_words.txt'))
            open(os.path.join(self._datadir, '.complete'), 'a').close()        
            
        self._synset_to_categorylist = {x.split(' ',1)[0]:[y.lstrip().rstrip() for y in x.split(' ', 1)[1].split(',')] for x in vipy.util.readtxt(os.path.join(self._datadir, 'synset_words.txt'))}            

        vipy.util.try_import('scipy.io', 'scipy')
        import scipy.io

        metadata = scipy.io.loadmat(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'), struct_as_record=False)
        synsets = np.squeeze(metadata['synsets'])
        ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
        wnids = np.squeeze(np.array([s.WNID for s in synsets]))
        words = np.squeeze(np.array([s.words for s in synsets]))
            
        self._wnid_to_categorylist = {wnid:[c.strip() for c in category.split(',')] for (wnid,category) in zip(wnids, words)}
                                    
        
    def synset_to_category(self, s=None):
        return self._wnid_to_categorylist if s is None else self._wnid_to_categorylist[s]
    
    def classification_trainset(self):
        """ImageNet2012 Classification, trainset"""
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_train'))  # slow-ish, may be better to cache
        imlist = ([vipy.image.TaggedImage(filename=f, attributes={'wordnet_id':f.rsplit('/',2)[-2]}, tags=self._wnid_to_categorylist[f.rsplit('/',2)[-2]]) for f in imgfiles] if os.name == 'posix' else
                  [vipy.image.TaggedImage(filename=f, attributes={'wordnet_id':os.path.basename(os.path.dirname(f))}, tags=self._wnid_to_categorylist[os.path.basename(os.path.dirname(f))]) for f in imgfiles])
        
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification:train')
        
    def classification_valset(self):
        """ImageNet2012 Classification, valset"""        
        imlist = []
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_val'))  # slow-ish, may be better to cache
                    
        # ground truth is imagenet synset index 1-1000
        gt = vipy.util.readtxt(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12', 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'))
        for (f,y) in zip(sorted(imgfiles), gt):
            imlist.append( (f, y) )

        # Index mapping is in mat file (yuck)
        synsets = self.synset_to_category()
        d_idx_to_category = {str(k):self.synset_to_category(r[0][1][0]) for (k,r) in enumerate(scipy.io.loadmat(os.path.join(self._datadir, 'ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat'))['synsets'], start=1) if r[0][1][0] in synsets}

        imlist = [vipy.image.TaggedImage(filename=f, attributes={'synset_index':y}, tags=d_idx_to_category[y]) for (f,y) in imlist] 
        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification:val')

    def faces(self):
        """Return all annotated faces in 2012 train and val sets:
        https://image-net.org/face-obfuscation/
        """

        cachefile = os.path.join(self._datadir, 'faces.json')
        if not os.path.exists(cachefile):        
            if not os.path.exists(os.path.join(self._datadir, 'face_annotations_ILSVRC.json')):
                vipy.downloader.download(IMAGENET_FACES, os.path.join(self._datadir, filetail(IMAGENET_FACES)))

            faces = {vipy.util.filetail(d['url']):d['bboxes'] for d in vipy.util.readjson(os.path.join(self._datadir, 'face_annotations_ILSVRC.json'))}
            imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_train'))  # slow-ish, may be better to cache
            imgfiles += vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2012_img_val'))  # slow-ish, may be better to cache
            
            imlist = [vipy.image.Scene(filename=f).objects([vipy.object.Detection(category='face', xmin=o['x0'], ymin=o['y0'], xmax=o['x1'], ymax=o['y1']) for o in (faces[filetail(f)] if filetail(f) in faces else [])]) for f in imgfiles]
            vipy.save(imlist, cachefile)
        else:
            try:
                imlist = vipy.load(cachefile)
            except:
                if os.path.exists(cachefile):
                    os.remove(cachefile)  # force recache on failure
                return self.faces()
            
        return vipy.dataset.Dataset([im for im in imlist if im.num_objects()>0], 'imagenet2012_faces:train')
        
    def localization_trainset(self):
        """ImageNet2012 localization, imageset = {train, val}, this takes a long time to read the XML files, load and cache"""

        cachefile = os.path.join(self._datadir, 'localization_trainset.json')
        if not os.path.exists(cachefile):
            log.warning('Initial parsing of XML files for imagenet-localization takes a long time... Consider "pip install lxml ujson" to speed this up')
        
            imlist = []
            classification = self.classification_trainset()
            synsets = self.synset_to_category()
            for f in classification._ds:                
                xmlfile = '%s.xml' % filefull(f.replace('ILSVRC2012_img_train', 'ILSVRC2012_bbox_train_v2'))
                im = vipy.image.Scene(filename=f)
                if os.path.exists(xmlfile):
                    root = ET.parse(xmlfile).getroot()  
                    for obj in root.findall("object"):
                        name = obj.findtext("name")
                        b = obj.find("bndbox")
                        xmin = int(b.findtext("xmin"))
                        ymin = int(b.findtext("ymin"))
                        xmax = int(b.findtext("xmax"))
                        ymax = int(b.findtext("ymax"))
                        im.add_object(vipy.object.Detection(tags=synsets[name] if name in synsets else [name], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
                if im.num_objects()>0:
                    imlist.append(im)                
            vipy.save(imlist, cachefile)  # cached
            
        else:
            try:
                imlist = vipy.load(cachefile)
            except:
                if os.path.exists(cachefile):
                    os.remove(cachefile)  # force recache on failure
                return self.localization_trainset()  
                               
        return vipy.dataset.Dataset(imlist, 'imagenet2012_localization:train')
    
                
class Imagenet21K_Resized(vipy.dataset.Dataset):
    """Imagenet21K_Resized requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term    
       https://image-net.org/download-images.php, imagenet-21K 2021 release ("squish" resized)"""
    def __init__(self, datadir=None, aslemma=True, redownload=False, recache=False):

        datadir = tocache('imagenet21k_resized') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)
        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.globals.log.info('[vipy.data.imagenet]: downloading Imagenet-21K resized to "%s"' % self._datadir)            
            vipy.downloader.download_and_unpack(IMAGENET21K_RESIZED_URL, self._datadir, sha1=None)
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt'))
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt'))

            open(os.path.join(self._datadir, '.complete'), 'a').close()
        
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

        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
        

class Imagenet21K(vipy.dataset.Dataset):
    """Imagenet21K requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term    
       imagenet-21K 2021 winter release

       https://image-net.org
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
                    vipy.downloader.unpack(f, filefull(f), progress=False)
                    os.remove(f)  # cleanup
                
            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt'))
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt'))

            open(os.path.join(self._datadir, '.complete'), 'a').close()
        
        cachefile = os.path.join(self._datadir, '.imlist.txt')
        if recache and os.path.exists(cachefile):
            os.remove(cachefile)
            
        # Class names: https://github.com/google-research/big_transfer/issues/7
        self._synset_to_categorylist = {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt')))}

        f_category = lambda c, synset_to_categorylist=self._synset_to_categorylist, aslemma=aslemma: synset_to_categorylist[c] if aslemma else c
        imglist = vipy.util.findimages(os.path.join(datadir, 'winter21_whole')) if not os.path.exists(cachefile) else vipy.util.readlist(cachefile)
        
        imlist = ([vipy.image.TaggedImage(filename=f, attributes={'wordnet_id':f.rsplit('/',2)[-2]}, tags=f_category(f.rsplit('/',2)[-2])) for f in imglist] if os.name == 'posix' else
                  [vipy.image.TaggedImage(filename=f, attributes={'wordnet_id':os.path.basename(os.path.dirname(f))}, tags=f_category(os.path.basename(os.path.dirname(f)))) for f in imglist])
        super().__init__(imlist, id='imagenet21k')

        if not os.path.exists(cachefile):
            vipy.util.writelist(imlist, cachefile)  # cache me for faster loading instead of walking the directory tree, not relocatable

        
    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
         
   
class Imagenet2014_DET():
    """Imagenet2014_DET requires login at https://image-net.org from the same IP address as the download, and agreeing to the ImageNet terms: https://image-net.org/download-images.php#term"""    
    def __init__(self, datadir=None, redownload=False, recache=False):    
        datadir = tocache('imagenet2014_det') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)

        if recache:
            if os.path.exists(os.path.join(self._datadir, 'imagenet2014_det_trainset.json')):
                os.remove(os.path.join(self._datadir, 'imagenet2014_det_trainset.json'))
            if os.path.exists(os.path.join(self._datadir, 'imagenet2014_det_valset.json')):
                os.remove(os.path.join(self._datadir, 'imagenet2014_det_valset.json'))
                
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            for url in URLS_2014:
                vipy.downloader.download(url, os.path.join(self._datadir, filetail(url)))

            vipy.downloader.download(IMAGENET21K_WORDNET_ID, os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt'))
            vipy.downloader.download(IMAGENET21K_WORDNET_LEMMAS, os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt'))
                
            for url in URLS_2014:
                vipy.downloader.unpack(os.path.join(self._datadir, filetail(url)), remkdir(os.path.join(self._datadir, filebase(url))))
        
            for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2014_DET_train')):
                vipy.downloader.unpack(f, filefull(f))

            for f in vipy.util.findtar(os.path.join(self._datadir, 'ILSVRC2014_DET_val')):
                vipy.downloader.unpack(f, filefull(f))

            for f in vipy.util.findtargz(os.path.join(self._datadir, 'bboxes_annotation')):
                vipy.downloader.unpack(f, filefull(f))
                
            open(os.path.join(self._datadir, '.complete'), 'a').close()        

    def synset_to_category(self):
        return {x:[y.rstrip().lstrip() for y in lemma.split(',')] for (x,lemma) in zip(vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_ids.txt')), vipy.util.readtxt(os.path.join(self._datadir, 'imagenet21k_wordnet_lemmas.txt')))}        
            
    def trainset(self):
        cachefile = os.path.join(self._datadir, 'imagenet2014_det_trainset.json')

        if not os.path.exists(cachefile):
            log.warning('Initial parsing of XML files takes a long time... Consider "pip install lxml ujson" to speed this up')            
            imgfiles = {filebase(f):f for f in vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2014_DET_train'))}  # slow-ish
            xmlfiles = {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'bboxes_annotation'))}  # slow-ish
            xmlfiles |= {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'ILSVRC2014_DET_bbox_train'))}  # slow-ish
            xmlfiles |= {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'ILSVRC2012_bbox_train_v2'))}  # slow-ish
            d_synset_to_category = self.synset_to_category()
            
            imlist = []        
            for (k,f) in imgfiles.items():
                im = vipy.image.Scene(filename=f)
                if k in xmlfiles and os.path.exists(xmlfiles[k]):
                    root = ET.parse(xmlfiles[k]).getroot()  
                    for obj in root.findall("object"):
                        name = obj.findtext("name")
                        subcategory = obj.findtext("name")                    
                        b = obj.find("bndbox")
                        xmin = int(b.findtext("xmin"))
                        ymin = int(b.findtext("ymin"))
                        xmax = int(b.findtext("xmax"))
                        ymax = int(b.findtext("ymax"))
                        tags = (d_synset_to_category[name] if name in d_synset_to_category else [name]) + (d_synset_to_category[subcategory] if subcategory in d_synset_to_category else [subcategory])
                        im.add_object(vipy.object.Detection(tags=tags, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax), attributes={'wordnet_id':name, 'subcategory':subcategory}))
                imlist.append(im)                
            vipy.save(imlist, cachefile)  # cached
        else:
            try:
                imlist = vipy.load(cachefile)
            except:
                if os.path.exists(cachefile):
                    os.remove(cachefile)  # force recache on failure
                return self.trainset()
            
        return vipy.dataset.Dataset(imlist, 'imagenet2014_det:trainset')


    def valset(self):
        cachefile = os.path.join(self._datadir, 'imagenet2014_det_valset.json')

        if not os.path.exists(cachefile):
            log.warning('Initial parsing of XML files takes a long time... Consider "pip install lxml ujson" to speed this up')            
            imgfiles = {filebase(f):f for f in vipy.util.findimages(os.path.join(self._datadir, 'ILSVRC2013_DET_val'))}  # slow-ish
            xmlfiles = {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'bboxes_annotation'))}  # slow-ish
            xmlfiles |= {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'ILSVRC2013_DET_bbox_val'))}  # slow-ish
            xmlfiles |= {filebase(f):f for f in vipy.util.findxml(os.path.join(self._datadir, 'ILSVRC2012_bbox_val_v3'))}  # slow-ish
            d_synset_to_category = self.synset_to_category()
            
            imlist = []        
            for (k,f) in imgfiles.items():
                im = vipy.image.Scene(filename=f)
                if k in xmlfiles and os.path.exists(xmlfiles[k]):
                    root = ET.parse(xmlfiles[k]).getroot()  
                    for obj in root.findall("object"):
                        name = obj.findtext("name")
                        subcategory = obj.findtext("name")                    
                        b = obj.find("bndbox")
                        xmin = int(b.findtext("xmin"))
                        ymin = int(b.findtext("ymin"))
                        xmax = int(b.findtext("xmax"))
                        ymax = int(b.findtext("ymax"))
                        tags = (d_synset_to_category[name] if name in d_synset_to_category else [name]) + (d_synset_to_category[subcategory] if subcategory in d_synset_to_category else [subcategory])
                        im.add_object(vipy.object.Detection(tags=tags, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax), attributes={'wordnet_id':name, 'subcategory':subcategory}))
                imlist.append(im)                
            vipy.save(imlist, cachefile)  # cached
        else:
            try:
                imlist = vipy.load(cachefile)
            except:
                if os.path.exists(cachefile):
                    os.remove(cachefile)  # force recache on failure
                return self.trainset()
            
        return vipy.dataset.Dataset(imlist, 'imagenet2014_det:valset')


    def testset(self):
        return vipy.dataset.Dataset.from_directory(os.path.join(self._datadir, 'ILSVRC2013_DET_test'), filetype='jpg').id('imagenet2014_det:testset')

