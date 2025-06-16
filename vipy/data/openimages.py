import os
import vipy
from vipy.util import filetail


def open_images_v7(datadir=None):
    """https://storage.googleapis.com/openimages/web/download_v7.html
     https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer
     https://github.com/Tencent/tencent-ml-images?tab=readme-ov-file#download-images

    ```python
     trainset = open_images_v7()
     d = trainset.take(16).map(lambda im: im.load(ignoreErrors=True)).filter(lambda im: im.isloaded())
     vipy.visualize.montage(d.map(lambda im: im.mindim(512).centersquare().annotate()).list()).show()
    ```

    This returns the tagged image and bounding boxes for the open_images v7 trainset
    
    This can take a long while to parse due to large csv files

    """

    urls = ['https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
            'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv',
            'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels.csv',    
            'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv',
            'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv',
            'https://storage.googleapis.com/openimages/v7/oidv7-classes-trainable.txt',
            'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv',
            'https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv',
            'https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv']
    
    labeldir = vipy.util.remkdir(vipy.util.tocache('open_images_v7') if not datadir else datadir)

    for url in urls:
        if not os.path.exists(os.path.join(labeldir, filetail(url))):
            vipy.downloader.download(url, os.path.join(labeldir, filetail(url)))
        
    trainable = set(vipy.util.readlist(os.path.join(labeldir, 'oidv7-classes-trainable.txt')))
    d_label_to_category = {r[0]:r[1] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-class-descriptions.csv'), ignoreheader=True)}
    d_label_to_category |= {r[0]:r[1] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-class-descriptions.csv'), ignoreheader=True)}
    d_label_to_object_category = {r[0]:r[1] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-class-descriptions-boxable.csv'), ignoreheader=True)}

    d_train_imageid_to_labels = vipy.util.groupbyasdict([(r[0], r[2]) for r in vipy.util.readcsv(os.path.join(labeldir, 'train-annotations-human-imagelabels.csv'), ignoreheader=True) if float(r[3])>0], keyfunc=lambda x: x[0], valuefunc=lambda x: x[1])
    d_train_imageid_to_objects = vipy.util.groupbyasdict([(r[0], r[2], float(r[4]), float(r[6]), float(r[5]), float(r[7]))  # (imageid, category, xmin, ymin, xmax, ymax)
                                                          for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-train-annotations-bbox.csv'), ignoreheader=True)], keyfunc=lambda x: x[0])
    d_train_imageid_to_url = {r[0]:r[2] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-train-images-with-labels-with-rotation.csv'), ignoreheader=True)}    
    d_train_url_to_category = {r[2]:[d_label_to_category[c] for c in d_train_imageid_to_labels[r[0]]]
                               for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-train-images-with-labels-with-rotation.csv'), ignoreheader=True) if r[0] in d_train_imageid_to_labels}

    #trainset = [(d_train_imageid_to_url[iid], [(d_label_to_object_category[o[1]], (o[2],o[3],o[4],o[5])) for o in obj if o[1] in d_label_to_object_category])
    #            for (iid, obj) in d_train_imageid_to_objects.items()]  # [(url, [(category, ulbr),...]), ...]    
    #loader = lambda r: vipy.image.Scene(url=r[0], tags=d_train_url_to_category[r[0]], objects=[vipy.object.Detection(category=c, ulbr=ulbr, normalized_coordinates=True) for (c,ulbr) in r[1]])
    #return vipy.dataset.Dataset(trainset, id='open_images_v7', loader=loader)
    
    imlist = [vipy.image.Scene(url=d_train_imageid_to_url[iid],
                               tags=d_train_url_to_category[d_train_imageid_to_url[iid]],
                               objects=[vipy.object.Detection(category=d_label_to_object_category[o[1]], ulbr=(o[2],o[3],o[4],o[5]), normalized_coordinates=True) for o in obj if o[1] in d_label_to_object_category])
              for (iid, obj) in d_train_imageid_to_objects.items()]  
    
    return vipy.dataset.Dataset(imlist, id='open_images_v7')

