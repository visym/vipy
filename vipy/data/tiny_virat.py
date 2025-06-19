import os
import vipy

    
class TinyVIRAT():
    """https://tinyactions-cvpr22.github.io/#dataset-details"""
    URL = 'https://www.crcv.ucf.edu/tiny-actions-challenge-cvpr2021/data/TinyVIRAT-v2.zip'
    def __init__(self, datadir):
        jsonfile = os.path.join(datadir, 'TinyVIRAT_V2','tiny_train_v2.json')
        if not os.path.exists(jsonfile):
            vipy.downloader.download_and_unpack(TinyVIRAT.URL, datadir)

        self._trainset = vipy.util.readjson(os.path.join(datadir, 'TinyVIRAT_V2','tiny_train_v2.json'))
        self._valset = vipy.util.readjson(os.path.join(datadir, 'TinyVIRAT_V2','tiny_val_v2.json'))
        self._testset = vipy.util.readjson(os.path.join(datadir, 'TinyVIRAT_V2','tiny_test_v2_public.json'))                
        self._datadir = datadir
        
    def trainset(self):
        return vipy.dataset.Dataset([vipy.video.Scene(tags=t['label'], filename=os.path.join(self._datadir, 'TinyVIRAT_V2', 'videos', 'train', t['path'])) for t in self._trainset], id='TinyVIRAT:train')

    def valset(self):
        return vipy.dataset.Dataset([vipy.video.Scene(tags=t['label'], filename=os.path.join(self._datadir, 'TinyVIRAT_V2', 'videos', 'val', t['path'])) for t in self._valset], id='TinyVIRAT:val')

    def testset(self):
        return vipy.dataset.Dataset([vipy.video.Video(filename=os.path.join(self._datadir, 'TinyVIRAT_V2', 'videos', 'test', t['path'])) for t in self._testset], id='TinyVIRAT:test')
    
