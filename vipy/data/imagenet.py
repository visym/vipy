import os
import vipy
from vipy.util import readcsv, remkdir, filepath, islist
from vipy.image import ImageDetection, ImageCategory


IMAGENET_21K_RESIZED_URL = 'https://image-net.org/data/imagenet21k_resized.tar.gz'
URLS_2012 = ['https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_dogs.tar.gz',
             'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz']
URL_SYNSET = 'https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt'

IMAGENET21K_WORDNET_ID = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt'
IMAGENET21K_WORDNET_LEMMAS = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'


class Imagenet2012(vipy.dataset.Dataset):
    """By downloading, you agree to the ImageNet terms: https://image-net.org/download-images.php#term"""
    def __init__(self, datadir):
        self._datadir = remkdir(datadir)

        for url in URLS_2012:
            if not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir)

        if not os.path.exists(os.path.join(self._datadir, 'synset_words.txt')):
            vipy.downloader.download(URL_SYNSET, os.path.join(self._datadir, 'synset_words.txt'))            
        self._synset_to_categorylist = {x.split(' ',1)[0]:[y.lstrip().rstrip() for y in x.split(' ', 1)[1].split(',')] for x in vipy.util.readtxt(os.path.join(self._datadir, 'synset_words.txt'))}            
        
    def __repr__(self):
        return str('<vipy.data.imagenet-2012: %s>' % self._datadir)

    def localization(self, imageset='train'):
        """ImageNet localization, imageset = {train, val}"""
        import xmltodict
        if imageset == 'train':
            imagesetfile = 'train_loc.txt'
        elif imageset == 'val':
            imagesetfile = 'val.txt'
        else:
            raise ValueError('unsupported imageset')

        csv = readcsv(os.path.join(self._datadir, 'ImageSets', 'CLS-LOC', imagesetfile), separator=' ')

        imlist = []
        for (path, k) in csv:
            xmlfile = '%s.xml' % os.path.join(self._datadir, 'Annotations', 'CLS-LOC', imageset, path)
            d = xmltodict.parse(open(xmlfile, 'r').read())
            imfile = '%s.JPEG' % os.path.join(self._datadir, 'Data', 'CLS-LOC', imageset, path)            
            imlist.append(vipy.image.Scene(filename=imfile, objects=[vipy.object.Detection(category=obj['name'],
                                                                                           xmin=int(obj['bndbox']['xmin']), ymin=int(obj['bndbox']['ymin']),
                                                                                           xmax=int(obj['bndbox']['xmax']), ymax=int(obj['bndbox']['ymax']))
                                                                   for obj in vipy.util.tolist(d['annotation']['object'])]))
                
        return vipy.dataset.Dataset(imlist, 'imagenet2012_localization_%s' % imageset)

    def synset_to_category(self, s=None):
        return self._synset_to_categorylist if s is None else self._synset_to_categorylist[s]
    
    def classes(self):
        return self.classification().categories()

    def classification(self, imageset='train'):
        """ImageNet Classification, imageset = {train, val}"""
        import xmltodict
        if imageset == 'train':
            imagesetfile = 'train_cls.txt'
        elif imageset == 'val':
            imagesetfile = 'val.txt'
        else:
            raise ValueError('unsupported imageset')
        csv = readcsv(os.path.join(self._datadir, 'ImageSets', 'CLS-LOC', imagesetfile), separator=' ')

        imlist = []
        for (subpath, k) in csv:
            xmlfile = '%s.xml' % os.path.join(self._datadir, 'Annotations', 'CLS-LOC', imageset, subpath)
            imfile = '%s.JPEG' % os.path.join(self._datadir, 'Data', 'CLS-LOC', imageset, subpath)
            if os.path.exists(xmlfile):
                d = xmltodict.parse(open(xmlfile, 'r').read())
                objlist = vipy.util.tolist(d['annotation']['object'])
                im = ImageCategory(filename=imfile, category=objlist[0]['name'])
            else:
                im = ImageCategory(filename=imfile, category=filepath(subpath))
            imlist.append(im)

        return vipy.dataset.Dataset(imlist, 'imagenet2012_classification_%s' % imageset)

                
class Imagenet21K(vipy.dataset.Dataset):
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
        
    
