import os
import vipy


TRAIN_IMG_2021_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz'
TRAIN_IMG_2021_MD5 = 'e0526d53c7f7b2e3167b2b43bb2690ed'
TRAIN_ANNO_2021_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz'
TRAIN_ANNO_2021_MD5 = '38a7bb733f7a09214d44293460ec0021'
VAL_IMG_2021_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz'
VAL_IMG_2021_MD5 = 'f6f6e0e242e3d4c9569ba56400938afc'
VAL_ANNO_2021_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz'
VAL_ANNO_2021_MD5 = '4d761e0f6a86cc63e8f7afc91f6a8f0b'


class iNaturalist2021(vipy.dataset.Dataset):
    """Project: https://github.com/visipedia/inat_comp/tree/master/2021"""
    def __init__(self, datadir=None, imageurl=TRAIN_IMG_2021_URL, imagemd5=TRAIN_IMG_2021_MD5, annourl=TRAIN_ANNO_2021_URL, annomd5=TRAIN_ANNO_2021_MD5, name='inaturalist', redownload=False):

        datadir = vipy.util.tocache('iNaturalist2021') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(imageurl, self._datadir, md5=imagemd5)
            vipy.downloader.download_and_unpack(annourl, self._datadir, md5=annomd5)

        json = vipy.util.readjson(os.path.join(self._datadir, '2021', vipy.util.filetail(annourl)[:-7]))  # remove trailing file extension (val.json.tar.gz -> val.json)

        d_imageid_to_filename = {x['id']:os.path.join(self._datadir, '2021', x['file_name']) for x in json['images']}
        d_imageid_to_annotation = {iid:a[0] for (iid,a) in vipy.util.groupbyasdict(json['annotations'], lambda x: x['image_id']).items()}  # one annotation per image
        d_categoryid_to_category = {x['id']:x['name'] for x in json['categories']}
        
        imlist = [(f,iid) for (iid,f) in d_imageid_to_filename.items()]

        loader = lambda x,d_imageid_to_annotation=d_imageid_to_annotation, d_categoryid_to_category=d_categoryid_to_category: vipy.image.ImageCategory(filename=x[0],
                                                    category=d_categoryid_to_category[d_imageid_to_annotation[x[1]]['category_id']] if x[1] in d_imageid_to_annotation else None,
                                                    attributes={'category_id': d_imageid_to_annotation[x[1]]['category_id']} if x[1] in d_imageid_to_annotation else None)
                                                    
        super().__init__(imlist, id=name, loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
    def trainset(self):
        return self.id('inaturalist:train')

    def valset(self):
        return iNaturalist2021(self._datadir, VAL_IMG_2021_URL, VAL_IMG_2021_MD5, VAL_ANNO_2021_URL, VAL_ANNO_2021_MD5, name='inaturalist:val')

    
