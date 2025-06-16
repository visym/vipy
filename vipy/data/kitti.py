import os
import vipy
import torchvision.datasets


class KITTI(vipy.dataset.Dataset):
    """A thin wrapper around torchvision.datasets to import into vipy.dataset format.  
    
    https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Kitti.html
    """
    
    def __init__(self, rootdir, download=False, split='train'):
        dataset = torchvision.datasets.Kitti(rootdir, download=download or not os.path.exists(os.path.join(rootdir, 'Kitti')), train=split=='train')

        loader = lambda r: (vipy.image.Scene(objects=[vipy.object.Detection(category=d['type'], ulbr=d['bbox'], attributes=d) for d in r[1]]) if r[1] is not None else vipy.image.Image()).loader(vipy.image.Image.PIL_loader, r[0])
        super().__init__(dataset, id='kitti:'+split, loader=loader)
        
