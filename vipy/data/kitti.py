import os
from vipy.dataset import Dataset
from vipy.image import Scene, Image
from vipy.util import try_import
from vipy.object import Detection


class KITTI(Dataset):
    """A thin wrapper around torchvision.datasets to import into vipy.dataset format.  
    
    https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Kitti.html
    """
    
    def __init__(self, rootdir, download=False, split='train'):
        try_import('torchvision.datasets', 'torchvision');
        import torchvision.datasets
        
        dataset = torchvision.datasets.Kitti(rootdir, download=download or not os.path.exists(os.path.join(rootdir, 'Kitti')), train=split=='train')

        loader = lambda r: (Scene(objects=[Detection(category=d['type'], ulbr=d['bbox'], attributes=d) for d in r[1]]) if r[1] is not None else Image()).loader(Image.PIL_loader, r[0])
        super().__init__(dataset, id='kitti:'+split, loader=loader)
        
