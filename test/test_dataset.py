import os
import numpy as np
import vipy.video
import vipy.videosearch
import vipy.object
from vipy.util import tempjpg, tempdir, Failed, isurl, rmdir, flatlist, to_iterable
from vipy.geometry import BoundingBox
from ipdb import set_trace
import warnings


def _test_registry(name=None):
    datasets = vipy.dataset.registry() if name is None else to_iterable(name)
    for d in datasets:
        vipy.dataset.registry(d)
        print('[test_registry]: %s  PASSED' % d)


def test_union():
    raw = ['d','a','c','a']
    raw_interleaved = [raw[0],raw[2],raw[1],raw[3]]
    D1 = vipy.dataset.Dataset([raw[0],raw[2]], id='test1')
    D2 = vipy.dataset.Dataset([raw[1],raw[3]], id='test2')
    D = D1 | D2

    assert raw == [x for x in D]
    print('[test_union]: __iter__  PASSED')

    assert raw == [D[k] for k in range(4)]
    print('[test_union]: __getitem__  PASSED')
    
    assert D.clone().sort(lambda x: x).list() == sorted(raw)
    print('[test_union]: sort  PASSED')

    assert D.clone().repeat(1).list() == raw*2
    print('[test_union]: repeat PASSED')

    assert D.count(lambda x: x=='a') == 2
    print('[test_union]: count  PASSED')

    assert len(D.clone().filter(lambda x: x=='a')) == 2
    print('[test_union]: filter  PASSED')

    assert len(D.clone().take(2)) == 2
    print('[test_union]: take  PASSED')

    assert D.clone().takeone() in raw
    print('[test_union]: takeone  PASSED')

    assert D.clone().load().list() == raw
    print('[test_union]: load  PASSED')

    assert [x for d in D.clone().minibatch(1) for x in d] == raw
    print('[test_union]: minibatch  PASSED')

    assert D.clone().map(lambda x: x+x).list() == [r+r for r in raw]
    print('[test_union]: map PASSED')

    assert D.clone().zip(raw).list() == list(zip(raw,raw))
    print('[test_union]: zip PASSED')

    
def test_dataset():
    raw = ['d','a','c','a']
    D = vipy.dataset.Dataset(raw, id='test')

    assert raw == [x for x in D]
    print('[test_dataset]: __iter__  PASSED')

    assert raw == [D[k] for k in range(len(D))]
    print('[test_dataset]: __getitem__  PASSED')
    
    assert D.clone().sort(lambda x: x).list() == sorted(raw)
    print('[test_dataset]: sort  PASSED')

    assert (D | D).list() == vipy.util.flatlist(zip(raw,raw))
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
    test_union()            
    test_dataset()
    _test_registry('mnist')

