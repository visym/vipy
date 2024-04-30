import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir
from vipy.geometry import BoundingBox
import pdb
from vipy.data.kinetics import Kinetics400, Kinetics600, Kinetics700
from vipy.data.activitynet import ActivityNet
from vipy.data.lfw import LFW
import warnings


def _test_dataset():
    rmdir('/tmp/lfw')
    d = LFW('/tmp/lfw')
    d[0].saveas('lfw.jpg')
    print('[test_datasets]: LFW PASSED')
    
    warnings.warn('these datasets are crufty and have many missing youtube videos')    
    rmdir('/tmp/kinetics')
    d = Kinetics400('/tmp/kinetics').valset()
    v = d[0].load(verbose=True)[0].resize(rows=256).saveas('/tmp/kinetics.jpg')
    v = d[1].download(verbose=True).save()    
    print('[test_datasets]:  Kinetics400 PASSED')
    
    d = Kinetics600('/tmp/kinetics').testset()
    v = d[1].load()[0].resize(rows=256).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics600 PASSED')
    
    d = Kinetics700('/tmp/kinetics').trainset()
    v = d[2].load()[0].rescale(0.5).saveas('kinetics.jpg')
    print('[test_datasets]:  Kinetics700 PASSED')
    
    rmdir('/tmp/activitynet')
    d = ActivityNet('/tmp/activitynet').dataset()
    v = d[0].load()
    print('[test_datasets]:  ActivityNet  PASSED')
    

    
if __name__ == "__main__":
    _test_dataset()    

