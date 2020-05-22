import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir
from vipy.geometry import BoundingBox
import pdb
from vipy.dataset.kinetics import Kinetics400, Kinetics600, Kinetics700
from vipy.dataset.activitynet import ActivityNet
from vipy.dataset.lfw import LFW


def _test_dataset():    
    rmdir('/tmp/kinetics')
    d = Kinetics400('/tmp/kinetics').download().valset()
    v = d[0].load(verbose=True)[0].resize(rows=256).saveas('/tmp/kinetics.jpg')
    v = d[1].download(verbose=True).save()    
    print('[test_datasets]:  Kinetics400 PASSED')
    
    d = Kinetics600('/tmp/kinetics').download().testset()
    v = d[1].load()[0].resize(rows=256).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics600 PASSED')
    
    d = Kinetics700('/tmp/kinetics').download().trainset()
    v = d[2].load()[0].rescale(0.5).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics700 PASSED')
    
    rmdir('/tmp/activitynet')
    d = ActivityNet('/tmp/activitynet').download().dataset()
    v = d[0].load()
    print('[test_datasets]:  ActivityNet  PASSED')
    
    rmdir('/tmp/lfw')
    d = LFW('/tmp/lfw').download().dataset()
    d[0].saveas('lfw.jpg')
    print('Video.datasets: PASSED')

    
if __name__ == "__main__":
    _test_dataset()    

