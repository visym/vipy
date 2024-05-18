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
     (trainset, valset, testset) = open_images_v7(labeldir='/path/to/downloaded.csv')
     d = trainset.take(128)
     vipy.visualize.montage(d.map(lambda im: im.load(ignoreErrors=True)).filter(lambda im: im.isloaded()).map(lambda im: im.centersquare()).list()).show()
    ```

    """
    
    labeldir = vipy.util.tocache('open_images_v7') if not datadir else datadir  

    trainable = set(vipy.util.readlist(os.path.join(labeldir, 'oidv7-classes-trainable.txt')))
    d_label_to_category = {r[0]:r[1] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-class-descriptions.csv'), ignoreheader=True)}
    d_category_to_label = {v:k for (k,v) in d_label_to_category.items()}

    d_imageid_to_label = {r[0]:r[2] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-train-annotations-human-imagelabels.csv'), ignoreheader=True)}
    d_val_imageid_to_label = {r[0]:r[2] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv7-val-annotations-human-imagelabels.csv'), ignoreheader=True)}    
    d_imageid_to_label.update(d_val_imageid_to_label)
    
    d_train_url_to_category = {r[2]:d_label_to_category[d_imageid_to_label[r[0]]] for r in vipy.util.readcsv(os.path.join(labeldir, 'oidv6-train-images-with-labels-with-rotation.csv'), ignoreheader=True)}
    d_val_url_to_category = {r[2]:d_label_to_category[d_imageid_to_label[r[0]]] for r in vipy.util.readcsv(os.path.join(labeldir, 'validation-images-with-rotation.csv'), ignoreheader=True)}    

    trainset = [(url,category) for (url, category) in d_train_url_to_category.items() if d_category_to_label[category] in trainable]
    valset = [(url,category) for (url, category) in d_val_url_to_category.items() if d_category_to_label[category] in trainable]
    
    loader = lambda r: vipy.image.ImageCategory(url=r[0], category=r[1])    
    return (vipy.dataset.Dataset(trainset, id='open_images_v7:train', loader=loader),
            vipy.dataset.Dataset(valset, id='open_images_v7:val', loader=loader))
