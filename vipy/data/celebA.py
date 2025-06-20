import os
import vipy


class CelebA(vipy.dataset.Dataset):
    """A thin wrapper around torchvision.datasets to import into vipy.dataset format.  
    
    https://docs.pytorch.org/vision/0.21/generated/torchvision.datasets.CelebA.html#torchvision.datasets.CelebA

    Requires gdown
    """
    
    def __init__(self, rootdir, download=False, split='train'):
        vipy.util.try_import('torchvision.datasets', 'torchvision');
        import torchvision.datasets
        
        dataset = torchvision.datasets.CelebA(rootdir, download=download or not os.path.exists(os.path.join(rootdir, 'celeba/img_align_celeba.zip')), split=split, target_type=['identity','bbox'])

        # Boxes are wrong: https://github.com/pytorch/vision/issues/9008 
        #loader = lambda r: (vipy.image.Scene(objects=[vipy.object.Detection(category=str(int(r[1][0])), xywh=list(r[1][1]))]) if r[1] is not None else vipy.image.Image()).loader(vipy.image.Image.PIL_loader, r[0])

        # Fallback on category only (there is no mapping from index string to celebrity name)
        loader = lambda r: (vipy.image.ImageCategory(category=str(int(r[1][0]))) if r[1] is not None else vipy.image.Image()).loader(vipy.image.Image.PIL_loader, r[0])     
        super().__init__(dataset, id='celebA:'+split, loader=loader)

        
