import os
import vipy
import numpy as np
import json
import PIL
import io


def open_images_v7(datadir=None):
    """https://storage.googleapis.com/openimages/web/download_v7.html
     https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer
     https://github.com/Tencent/tencent-ml-images?tab=readme-ov-file#download-images

    ```python
     (trainset, valset) = open_images_v7()
     d = trainset.take(128).map(lambda im: im.load(ignoreErrors=True)).filter(lambda im: im.isloaded())
     vipy.visualize.montage(d.map(lambda im: im.centersquare()).list()).show()
    ```

    """
    
    labeldir = vipy.util.tocache('open_images_v7') if not datadir else datadir  

    trainable = set(vipy.util.readlist(os.path.join(labeldir, 'oidv7-classes-trainable.txt')))
    d_label_to_category = {r[0]:r[1] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-class-descriptions.csv'), ignoreheader=True)}
    d_category_to_label = {v:k for (k,v) in d_label_to_category.items()}

    d_train_imageid_to_labels = vipy.util.groupbyasdict([(r[0], r[2]) for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-train-annotations-human-imagelabels.csv'), ignoreheader=True) if float(r[3])>0], keyfunc=lambda x: x[0], valuefunc=lambda x: x[1])
    d_val_imageid_to_labels = vipy.util.groupbyasdict([(r[0], r[2]) for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-val-annotations-human-imagelabels.csv'), ignoreheader=True) if float(r[3])>0], keyfunc=lambda x: x[0], valuefunc=lambda x: x[1])        
    d_imageid_to_labels = d_train_imageid_to_labels | d_val_imageid_to_labels
    
    d_train_url_to_category = {r[2]:[d_label_to_category[c] for c in d_imageid_to_labels[r[0]]] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-train-images-with-labels-with-rotation.csv'), ignoreheader=True) if r[0] in d_train_imageid_to_labels}
    d_val_url_to_category = {r[2]:[d_label_to_category[c] for c in d_imageid_to_labels[r[0]]] for r in vipy.util.readcsv(os.path.join(labeldir, 'validation-images-with-rotation.csv'), ignoreheader=True) if r[0] in d_val_imageid_to_labels}

    trainset = [(url,[c for c in category if d_category_to_label[c] in trainable]) for (url, category) in d_train_url_to_category.items()]
    valset = [(url,[c for c in category if d_category_to_label[c] in trainable]) for (url, category) in d_val_url_to_category.items()]
    
    loader = lambda r: vipy.image.ImageCategory(url=r[0], category=r[1])    
    return (vipy.dataset.Dataset(trainset, id='open_images_v7:train', loader=loader),
            vipy.dataset.Dataset(valset, id='open_images_v7:val', loader=loader))
