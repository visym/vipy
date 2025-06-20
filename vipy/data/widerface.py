import os
from vipy.dataset import Dataset
from vipy.image import Scene, Image
from vipy.util import try_import
from vipy.object import Detection


class WiderFace(Dataset):
    """A thin wrapper around torchvision.datasets to import into vipy.dataset format.  
    
    https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.WIDERFace.html#torchvision.datasets.WIDERFace

    Requires gdown
    """
    
    def __init__(self, rootdir, download=False, split='train'):
        try_import('torchvision.datasets', 'torchvision');
        import torchvision.datasets
        
        dataset = torchvision.datasets.WIDERFace(rootdir, download=download or not os.path.exists(os.path.join(rootdir, 'widerface', ' WIDER_train.zip')), split=split)

        loader = lambda r: (Scene(objects=[Detection(category='face', xywh=xywh) for xywh in r[1]['bbox']], attributes=r[1]) if r[1] is not None else Image()).loader(Image.PIL_loader, r[0])
        super().__init__(dataset, id='widerface:'+split, loader=loader)
        
