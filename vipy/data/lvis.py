import os
from vipy.util import remkdir, tocache, filetail
import vipy.downloader
import vipy.dataset
from vipy.image import Scene
from vipy.object import Detection

TRAIN_URL = 'https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip'
TRAIN_IMG_URL = 'http://images.cocodataset.org/zips/train2017.zip'
VAL_URL = 'https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip'
VAL_IMG_URL = 'http://images.cocodataset.org/zips/val2017.zip'


class LVIS():
    """https://www.lvisdataset.org"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('lvis') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(TRAIN_URL, self._datadir)
            vipy.downloader.download_and_unpack(TRAIN_IMG_URL, self._datadir)                        
            vipy.downloader.download_and_unpack(VAL_URL, self._datadir)
            vipy.downloader.download_and_unpack(VAL_IMG_URL, self._datadir)                        

            open(os.path.join(self._datadir, '.complete'), 'a').close()
        
    def trainset(self):
        d = vipy.util.readjson(os.path.join(self._datadir, 'lvis_v1_train.json'))

        d_imageid_to_annotations = vipy.util.groupbyasdict(d['annotations'], lambda a: a['image_id'])
        d_categoryid_to_category = {c['id']:c for c in d['categories']}
        
        images = [Scene(filename=os.path.join(self._datadir, 'train2017', filetail(i['coco_url'])),
                        url=i['coco_url'],
                        objects=[Detection(tags=d_categoryid_to_category[a['category_id']]['synonyms'], xywh=a['bbox'],
                                           attributes={'synset':d_categoryid_to_category[a['category_id']]['synset'], 'def':d_categoryid_to_category[a['category_id']]['def']})
                                 for a in d_imageid_to_annotations[i['id']]] if i['id'] in d_imageid_to_annotations else [])
                  for i in d['images']]
            
        return vipy.dataset.Dataset(images, id='lvis:train')
        
    def valset(self):
        d = vipy.util.readjson(os.path.join(self._datadir, 'lvis_v1_val.json'))

        d_imageid_to_annotations = vipy.util.groupbyasdict(d['annotations'], lambda a: a['image_id'])
        d_categoryid_to_category = {c['id']:c for c in d['categories']}
        
        images = [Scene(filename=os.path.join(self._datadir, 'val2017', filetail(i['coco_url'])),
                        url=i['coco_url'],
                        objects=[Detection(tags=d_categoryid_to_category[a['category_id']]['synonyms'], xywh=a['bbox'],
                                           attributes={'synset':d_categoryid_to_category[a['category_id']]['synset'], 'def':d_categoryid_to_category[a['category_id']]['def']})
                                 for a in d_imageid_to_annotations[i['id']]] if i['id'] in d_imageid_to_annotations else [])
                  for i in d['images']]
            
        return vipy.dataset.Dataset(images, id='lvis:val')

        
