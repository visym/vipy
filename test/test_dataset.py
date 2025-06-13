import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir, flatlist, to_iterable
from vipy.geometry import BoundingBox
import pdb
import warnings


def _test_registry(name=None):
    datasets = vipy.dataset.registry() if name is None else to_iterable(name)
    for d in datasets:
        vipy.dataset.registry(d)
        print('[test_registry]: %s  PASSED' % d)
    
    
def test_dataset():
    raw = ['d','a','c','a']
    D = vipy.dataset.Dataset(raw, id='test')

    assert D.clone().sort(lambda x: x).list() == sorted(raw)
    print('[test_dataset]: sort  PASSED')

    assert (D | D).list() == flatlist(zip(raw,raw))
    print('[test_dataset]: union  PASSED')

    assert D.clone().repeat(1).list() == raw+raw
    print('[test_dataset]: repeat PASSED')

    assert D.count(lambda x: x=='a') == 2
    print('[test_dataset]: count  PASSED')

    assert len(D.clone().filter(lambda x: x=='a')) == 2
    print('[test_dataset]: filter  PASSED')

    assert len(D.clone().take(2)) == 2
    print('[test_dataset]: take  PASSED')

    assert D.clone().takeone() in raw
    print('[test_dataset]: takeone  PASSED')

    assert D.clone().load().list() == raw
    print('[test_dataset]: load  PASSED')

    assert [x for d in D.clone().minibatch(1) for x in d] == raw
    print('[test_dataset]: minibatch  PASSED')

    assert D.clone().map(lambda x: x+x).list() == [r+r for r in raw]
    print('[test_dataset]: map PASSED')

    assert D.clone().zip(raw).list() == list(zip(raw,raw))
    print('[test_dataset]: zip PASSED')
            
    
if __name__ == "__main__":
    test_dataset()    
    _test_registry('mnist')

