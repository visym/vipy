import os
import vipy
import vipy.downloader


COCO_2014_IMAGE_URL = 'http://images.cocodataset.org/zips/train2014.zip'
COCO_2014_ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'


class Detection_TrainVal_2014(vipy.dataset.Dataset):
    """Project: https://cocodataset.org"""
    def __init__(self, outdir):
        self._outdir = vipy.util.remkdir(outdir)
        
        if not os.path.exists(os.path.join(self._outdir, 'train2014.zip')):
            print('[vipy.data.coco]: downloading COCO 2014 train/val images to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(COCO_2014_IMAGE_URL, self._outdir, sha1=None)

        if not os.path.exists(os.path.join(self._outdir, 'annotations_trainval2014.zip')):
            print('[vipy.data.coco]: downloading COCO 2014 train/val annotations to "%s"' % self._outdir)            
            vipy.downloader.download_and_unpack(COCO_2014_ANNO_URL, self._outdir, sha1=None)

        json = vipy.util.readjson(os.path.join(self._outdir, 'annotations', 'instances_train2014.json'))

        d_imageid_to_filename = {x['id']:os.path.join(self._outdir, 'train2014', x['file_name']) for x in json['images']}
        d_imageid_to_annotations = vipy.util.groupbyasdict(json['annotations'], lambda x: x['image_id'])
        d_categoryid_to_category = {x['id']:x['name'] for x in json['categories']}
        
        imlist = [vipy.image.Scene(filename=f,
                                   objects=[vipy.object.Detection(label=d_categoryid_to_category[x['category_id']], xywh=x['bbox'])
                                            for x in d_imageid_to_annotations[iid]] if iid in d_imageid_to_annotations else None)
                  for (iid,f) in d_imageid_to_filename.items()]
        
        super().__init__(imlist, id='coco_2014')

        
        
