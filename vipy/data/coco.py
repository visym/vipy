import os
import vipy
import vipy.downloader
from vipy.util import tocache


COCO_2014_IMAGE_URL = 'http://images.cocodataset.org/zips/train2014.zip'
COCO_2014_ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'


# https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
labels_2014_2017 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Common issues:
# - comprehensive labeling of all books, oranges, broccoli, people, vases, boats in image
# - multiple books are often merged into "shelves" of books (/COCO_train2014_000000070000.jpg)
# - crowds are missing people (COCO_train2014_000000043971.jpg), or crowd is grouped as single person (COCO_train2014_000000510239.jpg)
# - parking lots missing vehicles (COCO_train2014_000000083770.jpg)
# - backpack/handbag with only strap visible
# - bed annotation with/without headboard
# - chairs and tables may be heavily occluded


class COCO_2014(vipy.dataset.Dataset):
    """Project: https://cocodataset.org Detection_Train_Val_2014"""
    def __init__(self, datadir=None, redownload=False):

        outdir = tocache('coco2014') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(outdir)
        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            print('[vipy.data.coco]: downloading COCO 2014 train/val images to "%s"' % self._datadir)            
            vipy.downloader.download_and_unpack(COCO_2014_IMAGE_URL, self._datadir, sha1=None)

            print('[vipy.data.coco]: downloading COCO 2014 train/val annotations to "%s"' % self._datadir)            
            vipy.downloader.download_and_unpack(COCO_2014_ANNO_URL, self._datadir, sha1=None)

        json = vipy.util.readjson(os.path.join(self._datadir, 'annotations', 'instances_train2014.json'))

        d_imageid_to_filename = {x['id']:os.path.join(self._datadir, 'train2014', x['file_name']) for x in json['images']}
        d_imageid_to_annotations = vipy.util.groupbyasdict(json['annotations'], lambda x: x['image_id'])
        d_categoryid_to_category = {x['id']:x['name'] for x in json['categories']}
        
        imtuple = tuple((f,iid) for (iid,f) in d_imageid_to_filename.items())
        loader = lambda x, d_categoryid_to_category=d_categoryid_to_category, d_imageid_to_annotations=d_imageid_to_annotations: vipy.image.Scene(filename=x[0],
                                                                                                                                                  objects=[vipy.object.Detection(category=d_categoryid_to_category[o['category_id']], xywh=o['bbox'])
                                                                                                                                                           for o in d_imageid_to_annotations[x[1]]] if x[1] in d_imageid_to_annotations else None)
        super().__init__(imtuple, id='coco_2014', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()        
        
