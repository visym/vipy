import vipy
import numpy as np
import copy
import os
import random
import dill
import time
import json
from vipy.globals import log

import vipy.util
vipy.util.try_import('torch');

import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split


class TorchDataset(torch.utils.data.Dataset):
    """Converter from a pycollector dataset to a torch dataset"""
    def __init__(self, f_transformer, d):
        import vipy.dataset
        assert isinstance(d, vipy.dataset.Dataset), "Invalid input"
        assert callable(f_transformer), "Invalid input"
        self._f_transformer = dill.dumps(f_transformer)  # for torch serialization of lambda functions        
        self.dataset = d
        
    def _unpack(self):
        if isinstance(self._f_transformer, bytes):
            self._f_transformer = dill.loads(self._f_transformer)        
        return self

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]
            
    def __getitem__(self, k):
        """Should return tuple(tensor, index)"""
        return self._unpack()._f_transformer(self.dataset[k])

    def __len__(self):
        return len(self.dataset)


class Tensordir(torch.utils.data.Dataset):
    """A torch dataset stored as a directory of .pkl.bz2 files each containing a list of [(tensor, str=json.dumps(label)), ...] tuples used for data augmented training.
    
    This is useful to use the default Dataset loaders in Torch.
    
    Usage:

    ```python
    vipy.torch.Tensordir('/path/to')
    vipy.torch.Tensordir( ('/path/to/1', '/path/to/2') )
    ```
    .. note:: This requires python random() and not numpy random 
    """
    def __init__(self, tensordir, verbose=True, reseed=True, take=None, mutator=None):
        assert (isinstance(tensordir, str) and os.path.isdir(tensordir)) or all([os.path.isdir(d) for d in tensordir])
        assert mutator is None or callable(mutator)

        self._dirlist = [s for d in vipy.util.tolist(tensordir) for s in vipy.util.extlist(d, '.pkl.bz2')]
        self._verbose = verbose
        self._reseed = reseed
        self._mutator = mutator 

    def __getitem__(self, k):
        if self._reseed:
            random.seed()  # force randomness after fork()

        assert k >= 0 and k < len(self._dirlist)
        for j in range(0,3):
            try:
                obj = vipy.util.bz2pkl(self._dirlist[k])  # load me
                assert len(obj) > 0, "Invalid augmentation"
                (t, lbl) = obj[random.randint(0, len(obj)-1)]  # choose one tensor at random
                assert t is not None and json.loads(lbl) is not None, "Invalid augmentation"  # get another one if the augmentation was invalid
                return (t, lbl if self._mutator is None else json.dumps(self._mutator(json.loads(lbl))))
            except:
                time.sleep(1)  # try again after a bit if another process is augmenting this .pkl.bz2 in parallel
        if self._verbose:
            log.warning('[vipy.dataset.TorchTensordir]: %s corrupted or invalid' % self._dirlist[k])
        return self.__getitem__(random.randint(0, len(self)-1))  # maximum retries reached, get another one

    def __len__(self):
        return len(self._dirlist)

    def take(self, n):
        self._dirlist = [self._dirlist[k] for k in np.random.permutation(range(len(self._dirlist)))[0:n]]
        return self

    def filter(self, f):
        """Keep elements that lambda evaluates true. The lambda operates on the *absolute path filename* for the tensordir and not the contents.  This is useful for filtering by instanceid in the `vipy.util.filebase`."""
        assert callable(f)
        self._dirlist = [x for x in self._dirlist if f(x)]
        return self
    
    def clone(self):
        return copy.deepcopy(self)


class TorchTensordir(Tensordir):
    pass  # alias for backwards compatibility
