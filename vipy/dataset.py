import os
import numpy as np
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml
from vipy.util import listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv
from vipy.util import newpathroot, listjson, extlist, filefull, tempdir, groupbyasdict, try_import, shufflelist
import random
import vipy
import vipy.util
import shutil
import uuid
import warnings
import copy 
from vipy.util import is_email_address
import hashlib
import pickle
import time
import json
import dill
from vipy.show import colorlist
import matplotlib.pyplot as plt
import gc 
import vipy.metrics
import itertools


class Dataset():
    """vipy.dataset.Dataset() class
    
    Common class to manipulate large sets of objects in parallel

    Args:
        - dataset [list, tuple, set, obj]: a python built-in type that supports indexing or a generic object that supports indexing and has a length
        - loader [lambda]: a callable loader that will construct the object from a raw data representation.  This is useful for custom deerialization or on demand transformations
        - strict [bool]: If true, throw an error if the type of objlist is not a python built-in type.  This is useful for loading dataset objects that can be indexed.
        - preprocessor [lambda]:  a callable preprocessing function that will preprocess the object. This is useful for implementing on-demand data loaders
        - index [list]: If provided, use this as the initial index into the dataset.  This is useful for preprocessing large datasets to filter out noise.
        - repeat [int]: Repeat the dataset.  If repeat=0, then there are no repeats. This is useful for generating random preprocessed samples of the same source data.  Repeated datasets are shared, with appended indexes
    """

    def __init__(self, dataset, id=None, loader=None, strict=True, preprocessor=None, shuffler=None, index=None, repeat=0):
        assert loader is None or callable(loader)
        assert preprocessor is None or callable(preprocessor)
        assert shuffler is None or callable(shuffler)        
        assert index is None or isinstance(index, (list, tuple))
        assert repeat >= 0
        
        self._id = id
        self._ds = dataset if not isinstance(dataset, (list, set, tuple)) else tuple(dataset)  # force immutable (if possible)
        self._idx = list(range(len(self._ds)) if not index else index)*(repeat+1)
        self._loader = loader  # not serializable if lambda is provided
        self._preprocessor = preprocessor
        self._shuffler = shuffler
        self._type = None
        
        assert not strict or index is None or (len(index)>0 and len(index)<=len(dataset) and max(index)<len(dataset) and min(index)>0)

            
    def __or__(self, other):
        assert isinstance(other, Dataset)
        return Union((self, other), id=self.id())

    
    def id(self, n=None, truncated=False, maxlen=80):
        """Set or return the dataset id, useful for showing the name/split of the dataset in the representation string"""
        if n is None:
            return (self._id[0:maxlex] + ' ... ') if truncated and self._id and len(self._id)>maxlen else self._id
        else:
            self._id = n
            return self

    def index(self, index=None):
        """Update the index, useful for filtering of large datasets"""
        if index is not None:
            self._idx = index
            return self
        return self._idx
    
    def raw(self):
        """Remove the loader and preprocessor, useful for cloned direct access of raw data in large datasets without loading every one"""
        self._loader = None
        self._preprocessor = None
        self._type = None
        return self

    def type(self):
        if self._type is None and len(self)>0:
            self._type = str(type(self[0]))  # peek at first element, cached
        return self._type
        
    def preprocessor(self, f=None, remove=False):
        if f is not None:
            assert callable(f)
            self._preprocessor = f
            self._type = None
            return self
        if remove:
            self._preprocessor = None
            self._type = None            
            return self
        return self._preprocessor

    def shuffler(self, f=None, remove=False):
        if f is not None:
            assert callable(f)
            self._shuffler = f
            return self
        if remove:
            self._shuffler = None
            return self
        return self._shuffler

    
    def __repr__(self):
        fields = ['id=%s' % self.id(truncated=True, maxlen=80)] if self.id() else []
        fields += ['len=%d' % len(self)]
        fields += ['type=%s' % self.type()] if self.type() else []
        return str('<vipy.dataset.%s: %s>' % (self.__class__.__name__, ', '.join(fields)))

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, k):
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(self._idx), "invalid index"
            x = self._ds[self._idx[int(k)]]
            x = self._loader(x) if self._loader is not None else x
            x = self._preprocessor(x) if self._preprocessor is not None else x
            return x
        elif isinstance(k, slice):
            X = [self._ds[k] for k in self._idx[k.start:k.stop:k.step]]
            X = [self._loader(x) for x in X] if self._loader is not None else X
            X = [self._preprocessor(x) for x in X] if self._preprocessor is not None else X
            return X
        else:
            raise ValueError('invalid index type "%s"' % type(k))            
            
    def __len__(self):
        return len(self._idx)

    def clone(self, shallow=False):
        """Return a copy of the dataset object"""
        if shallow:
            (idx, ds) = (self._idx, self._ds)
            (self._idx, self._ds) = ([], None)  # remove
            D = copy.copy(self) 
            (self._idx, self._ds) = (idx, ds)   # restore            
            (D._idx, D._ds)  = (idx, ds)  # shared index/object reference
            return D
        else:
            return copy.deepcopy(self)
    
    def shuffleif(self, b):
        return self.shuffle() if b else self
    
    def shuffle(self):
        """Permute elements in this dataset uniformly at random in place"""
        return Dataset.uniform_shuffler(self) if not self._shuffler else self._shuffler(self)

    def instanceid(self, k):
        """The instance ID of element k is the index of this instance in the underlying source dataset"""
        return self._idx[k]
    
    def repeat(self, n):
        """Repeat the dataset n times.  If n=0, the dataset is unchanged, if n=1 the dataset is doubled in length, etc."""
        assert n>=0
        self._idx = self._idx*(n+1)
        return self
    
    def list(self, mapper=None, reducer=None):
        """Return the dataset as a list, loading the entire dataset into memory, applying the optional mapper lambda on each element, and applying the optional reducer lambda on the resulting list"""
        assert mapper is None or callable(mapper)
        assert reducer is None or callable(reducer)
        mapped = [mapper(x) if mapper else x for x in self]
        reduced = reducer(mapped) if reducer else mapped
        return reduced

    def flatten(self):
        return Dataset(self.flatlist(), id=self.id())
        
    def flatlist(self, mapper=None):
        """Return the dataset as a flat list, converting lists of lists into a single flat list"""
        return self.list(mapper, reducer=lambda mapped: [x for xiter in mapped for x in xiter])
    
    def set(self, mapper):
        """Return the dataset as a set.  Mapper must be a lambda function that returns a hashable type"""
        assert callable(mapper)
        return {mapper(x) for x in self}
        

    def frequency(self, f):
        """Frequency counts for which lamba returns the same value"""
        assert callable(f)
        return vipy.util.countby(self, f)

    def count(self, f):
        """Counts for each element for which lamba returns true.  
        
        Args:
            f: [lambda] if provided, count the number of elements that return true.  This is the same as len(self.filter(f)) without modifying the dataset.

        Returns:
            A dictionary of counts per category [if f is None]
            A length of elements that satisfy f(v) = True [if f is not None]
        """
        assert callable(f)
        return len([k for (j,k) in enumerate(self._idx) if f(self[j])])
        
    def filter(self, f):
        """In place filter with lambda function f, keeping those elements obj in-place where f(obj) evaluates true"""
        assert callable(f)
        self._idx = [i for (b,i) in zip(self.map(f, ordered=True), self._idx) if b]        
        return self
    
    def take(self, n, inplace=False):
        """Randomly Take n elements from the dataset, and return a dataset (in-place or cloned)."""
        assert isinstance(n, int) and n>0
        D = self.clone(shallow=True) if not inplace else self
        D._idx = shufflelist(self._idx)[0:n]  # do not run loader, seed controlled by random.seed()
        return D

    def groupby(self, f):
        """Group the dataset according to the callable f, returning dictionary of grouped datasets."""
        assert callable(f)        
        return {k:self.clone(shallow=True).index([x[1] for x in v]).id('%s:%s' % (self.id(),str(k))) for (k,v) in itertools.groupby(enumerate(self.sort(f)._idx), lambda x: f(self[x[0]]))}

    def takeby(self, f, n):
        """Group the dataset according to the callable f, take n from each group and return a dictionary of lists"""
        return {k:v.takelist(n) for (k,v) in self.groupby(f).items()}
    
    def takelist(self, n):
        """Take n elements and return list.  The elements are loaded and not cloned."""
        return self.take(n).list()

    def takeone(self):
        """Randomly take one element from the dataset and return a singleton"""
        return self.takelist(n=1)[0] if len(self)>0 else None

    def sample(self):
        return self.takeone()
    
    def take_fraction(self, p, inplace=False):
        """Randomly take a percentage of the dataset, returning a clone or in-place"""
        assert p>=0 and p<=1
        return self.take(n=int(len(self)*p), inplace=inplace)
    
    def load(self):
        """Load the entire dataset into memory.  This is useful for creating in-memory datasets from lazy load datasets"""
        return Dataset(self.list(), id=self.id())
    
    def chunk(self, n):
        """Yield n chunks as dataset.  Last chunk will be ragged.  Batches are not loaded or preprocessed"""
        for (k,V) in enumerate(vipy.util.chunkgen(self._idx, n)):
            yield self.clone(shallow=True).index(V).id(('%s:%d' % (self.id(), k)) if self.id() else str(k))

    def batch(self, n):
        """Yield batches of size n as datasets.  Last batch will be ragged.  Batches are not loaded or preprocessed"""
        for (k,V) in enumerate(vipy.util.chunkgenbysize(self._idx, n)):
            yield self.clone(shallow=True).index(V).id(('%s:%d' % (self.id(), k)) if self.id() else str(k))
            
    def minibatch(self, n, ragged=True, concurrent=True):
        """Yield preprocessed minibatches of size n of this dataset.

        To yield chunks of this dataset, suitable for minibatch training/testing

        ```python
        D = vipy.dataset.Dataset(...)
        for b in D.minibatch(n):
           print(b)
        ```
        
        To perform minibatch preprocessing in parallel across four processes with the context manager:

        ```python
        D = vipy.dataset.Dataset(...)
        with vipy.globals.parallel(4):
            for b in D.minibatch(n):
                print(b)
        ```

        To perform minibatch preprocessing in parallel across four processes with distributed client:

        ```python
        D = vipy.dataset.Dataset(...)
        vipy.globals.parallel(4)    
        for b in D.minibatch(n):
           print(b)
        ```
        
        Args:
            n [int]: The size of the minibatch
            ragged [bool]: If ragged=true, then the last chunk will be ragged with len(chunk)<n, else skipped
            concurrent: If true and there exists a vipy.parallel.exeuctor(), load and preprocess minibatches in parallel 

        Returns:        
            Iterator over `vipy.dataset.Dataset` elements of length n.  Minibatches will be yielded loaded and preprocessed (processing done concurrently)

        ..note:: The distributed iterator appends the minibatch index to the minibatch.id()
        """
        mapper = vipy.parallel.map if concurrent and vipy.parallel.executor() else vipy.parallel.localmap
        for b in mapper(lambda b: b.load(), self.batch(n)):  # parallel load()
            if ragged or len(b) == n:            
                yield b

    def shift(self, m):
        """Circular shift the dataset m elements to the left, so that self[k+m] == self.shift(m)[k].  Circular shift for boundary handling so that self.shift(m)[-1] == self[m-1]"""
        return self.clone(shallow=True).index(self._idx[m:] + self._idx[0:m])

    def slice(self, start=0, stop=-1, step=1):
        """Slice the dataset to contain elements defined by slice(start, stop, step)"""
        return self.clone(shallow=True).index(self._idx[start:stop:step])
        
    def truncate(self, m):
        """Truncate the dataset to contain the first m elements only"""
        return self.slice(stop=m)
    
    def pipeline(self, n, m, ragged=True, concurrent=True):
        """Yield pipelined minibatches of size n with pipeline length m.

        A pipelined minibatch is a tuple (head, tail) such that (head, tail) are minibatches at different indexes in the dataset.  
        Head corresponds to the current minibatch and tail corresponds to the minibatch left shifted by (m-1) minibatches.

        This structure is useful for yielding datasets for pipelined training where head contains the minibatch that will complete pipeline training on this iteration, and tail contains the 
        next minibatch to be inserted into the pipeline on this iteration.
        
        ```python
        D = vipy.dataset.Dataset(...)
        for (head, tail) in D.pipeline(n, m):
            assert head == D[0:m]
            assert tail == D[n*(m-1): n*(m-1)+n]

        Args:
            n [int]: The size of each minibatch
            m [int]:  The pipeline length in minibatches
            ragged [bool]: If ragged=true, then the last chunk will be ragged with len(chunk)<n, else skipped
            distributed: If true, preprocess minibatches in parallel 

        Returns:        
            Iterator over tuples (head,tail) of `vipy.dataset.Dataset` elements of length n where tail is left shifted by n*(m-1) elements.  Minibatches will be preprocessed if distributed=True, else not-preprocessed
        
        .. note::  The distributed iterator is not order preserving over minibatches and yields minibatches as completed, however the tuple (head, tail) is order preserving within the pipeline
        
        """
        pipeline = list(self.truncate(n*(m-1)).minibatch(n, concurrent=False))  # local pipeline fill
        for b in self.shift(n*(m-1)).minibatch(n, ragged=ragged, concurrent=concurrent):  # not order preserving
            pipeline.append(b)  # order preserving within pipeline            
            yield( (pipeline.pop(0), b) )  # yield deque-like (minibatch, shifted minibatch) tuples
        
        
    def split(self, trainfraction=0.9, valfraction=0.1, testfraction=0, trainsuffix=':train', valsuffix=':val', testsuffix=':test'):
        """Split the dataset into the requested fractions.  

        Args:
            trainfraction [float]: fraction of dataset for training set
            valfraction [float]: fraction of dataset for validation set
            testfraction [float]: fraction of dataset for test set
            trainsuffix: If not None, append this string the to trainset ID
            valsuffix: If not None, append this string the to valset ID
            testsuffix: If not None, append this string the to testset ID        
        
        Returns:        
            (trainset, valset, testset) 
        """
        assert trainfraction >=0 and trainfraction <= 1
        assert valfraction >=0 and valfraction <= 1
        assert testfraction >=0 and testfraction <= 1
        assert abs(trainfraction + valfraction + testfraction - 1) < 1E-6
        
        idx = vipy.util.shufflelist(self._idx)
        (testidx, validx, trainidx) = vipy.util.dividelist(idx, (testfraction, valfraction, trainfraction))
            
        trainset = self.clone(shallow=True).index(trainidx)
        if trainsuffix and trainset.id():
            trainset.id(trainset.id() + trainsuffix)
        
        valset = self.clone(shallow=True).index(validx)
        if valsuffix and valset.id():
            valset.id(valset.id() + valsuffix)
        
        testset = self.clone(shallow=True).index(testidx)
        if testsuffix and testset.id():
            testset.id(testset.id() + testsuffix)
                
        return (trainset,valset,testset)

    def map(self, f_map, distributed=True, strict=True, ordered=False, oneway=False):        
        """Distributed map.

        To perform this in parallel across four processes:

        ```python
        D = vipy.dataset.Dataset(...)
        with vipy.globals.parallel(4):
            D.map(lambda v: ...)
        ```

        Args:
            f_map: [lambda] The lambda function to apply in parallel to all elements in the dataset.  This must return a JSON serializable object (or set oneway=True)
            distributed [bool]: If true and vipy.globals.dask() is not None, distribute map over workers, else local map only
            strict: [bool] If true, raise exception on map failures, otherwise the map will return only those that succeeded
            ordered: [bool] If true, preserve the order of objects in dataset as returned from distributed processing
            oneway: [bool] If true, do not pass back results unless exception
        
        Returns:
            A `vipy.dataset.Dataset` containing the elements f_map(v).  This operation is order preserving if ordered=True.

        .. note:: 
            - This method uses dask distributed and `vipy.batch.Batch` operations
            - Due to chunking, all error handling is caught by this method.  Use `vipy.batch.Batch` to leverage dask distributed futures error handling.
            - Operations must be chunked and serialized because each dask task comes with overhead, and lots of small tasks violates best practices
            - Serialized results are deserialized by the client and returned a a new dataset
        """
        assert callable(f_map), "invalid map function"        

        # Local map
        if not distributed or vipy.globals.dask() is None:            
            return self.localmap(f_map)
                    
        # Distributed map
        from vipy.batch import Batch   # requires pip install vipy[all]                
        f_serialize = lambda x: x
        f_deserialize = lambda x: x
        f_oneway = lambda x, oneway=oneway: x if not x[0] or not oneway else (x[0], None)
        f_catcher = lambda f, *args, **kwargs: vipy.util.catcher(f, *args, **kwargs)  # catch exceptions when executing lambda, return (True, result) or (False, exception)
        f = lambda x, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher, f_oneway=f_oneway: f_serializer(f_oneway(f_catcher(f_map, f_deserializer(x))))  # with closure capture

        S = [f_serialize(v) for v in self]  # local load, preprocess and serialize
        B = Batch(vipy.util.chunklist(S, 128), strict=False, warnme=False, minscatter=128, ordered=ordered)
        S = B.map(lambda X,f=f: [f(x) for x in X]).result()  # chunked, with caught exceptions, may return empty list
        V = [f_deserialize(x) for s in S for x in s]  # Local deserialization and chunk flattening
        
        # Error handling
        (good, bad) = ([r for (b,r) in V if b], [r for (b,r) in V if not b])  # catcher returns (True, result) or (False, exception string)
        if strict and len(bad)>0:
            raise ValueError('[vipy.dataset.Dataset.map]: Exceptions in distributed processing:\n%s\n\n[vipy.dataset.Dataset.map]: %d/%d items failed' % (str(bad), len(bad), len(self)))
        return Dataset(good, id=self.id()) if not oneway else None

    def localmap(self, f):
        return Dataset(self.list(f), id=self.id())  # triggers load into memory        

    def mapby_minibatch(self, f, n, ragged=True):
        return Dataset([f(b) for b in self.minibatch(n, ragged)], id=self.id())
    
    def sort(self, f):
        """Sort the dataset in-place using the sortkey lambda function f which is called either with one argument f(instance) or two arguments f(instance, instanceid)

        To perform a sort of the dataset using some property of the instance, such as the object category (e.g. for vipy.image objects) 

        ```python
        dataset.sort(lambda im: im.category())
        ```

        To sort the dataset back into the original order by instance id:
        
        ```python
        dataset.sort(lambda im, iid: iid)
        ```

        To perform a lexicographic sort of the instance id and the object category

        ```python
        dataset.sort(lambda iid, im: (im.category(), iid))
        ```

        """
        assert callable(f) and len(f.__code__.co_varnames) in [1,2]
        g = (lambda x,y: f(x)) if len(f.__code__.co_varnames)==1 else (lambda x,y: f(x,y)) 
        self._idx = [self._idx[j] for j in sorted(range(len(self)), key=lambda k: g(self[k], self._idx[k]))]
        return self

    def dedupe(self, f):
        """Deduplicate the dataset using the key lambda function f which is called either with one argument f(instance) or two arguments f(instance, instanceid)  

        To deduplicate by instance id:

        ```python
        dataset.dedupe(lambda im, iid: iid)
        ```

        To deduplicate by identical percetual hash (e.g. for vipy.image objects) with 10 threads:

        ```python
        with vipy.globals.parallel(workers=10):
            dataset.dedupe(lambda im: im.perceptualhash())
        ```

        """
        assert callable(f) and len(f.__code__.co_varnames) in [1,2]
        g = (lambda z: f(z[0])) if len(f.__code__.co_varnames)==1 else (lambda z: f(z[0],z[1]))  # unpack lambda tuple args
        self._idx = list({m:i for (i,m) in zip(self._idx, vipy.parallel.map(g, zip(self, self._idx)))}.values())
        return self
        
    @staticmethod
    def uniform_shuffler(D):
        random.shuffle(D._idx)        
        return D
    
    @staticmethod
    def identity_shuffler(D):
        """Shuffler that does nothing"""
        return D

    @staticmethod
    def chunk_shuffler(D, chunksize=64):
        """Split dataset into len(D)/chunksize non-overlapping chunks, shuffle chunk order and shuffle within chunks.  

           - This preserves microbatch neighbors when chunksize=batchsize
           - If chunksize=1 then this is equivalent to uniform_shuffler
        """        
        return D.index([i for I in shufflelist([shufflelist(I) for I in vipy.util.chunkgenbysize(D._idx, chunksize)]) for i in I])
    
    
class Paged(Dataset):
    """ Paged dataset.

    A paged dataset is a dataset of length N=M*P constructed from M archive files (the pages) each containing P elements (the pagesize).  
    The paged dataset must be constructed with tuples of (pagesize, filename).  
    The loader will fetch, load and cache the pages on demand using the loader, preserving the most recently used cachesize pages

    ```python
    D = vipy.dataset.Paged([(64, 'archive1.pkl'), (64, 'archive2.pkl')], lambda x,y: ivy.load(y))
    ```

    .. note :: Shuffling this dataset is biased.  Shuffling will be performed to mix the indexes, but not uniformly at random.  The goal is to preserve data locality to minimize cache misses.
    """
    
    def __init__(self, pagelist, loader, id=None, strict=True, preprocessor=None, index=None, cachesize=32, shuffler=None):        
        super().__init__(dataset=pagelist,
                         id=id,
                         loader=loader,
                         strict=False,
                         preprocessor=preprocessor,
                         index=index if index else list(range(sum([p[0] for p in pagelist]))),
                         shuffler=shuffler)

        assert callable(loader), "loader required"
        assert not strict or len(set([x[0] for x in self._ds])) == 1  # pagesizes all the same 

        self._cachesize = cachesize
        self._pagecache = {}
        self._ds = list(self._ds)
        self._pagesize = self._ds[0][0]  # (pagesize, pklfile) tuples        

        # Shuffle while preserve page locality to minimize cache misses
        self._shuffler = shuffler if shuffler else lambda D, chunksize=int(1.5*self._pagesize): Dataset.chunk_shuffler(D, chunksize)

        
    def __getitem__(self, k):
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(self._idx), "invalid index"
            page = self._idx[int(k)] // self._pagesize
            if page not in self._pagecache:
                self._pagecache[page] = self._loader(*self._ds[page])  # load and cache new page
                if len(self._pagecache) > self._cachesize:
                    self._pagecache.pop(list(self._pagecache.keys())[0])  # remove oldest
            x = self._pagecache[page][int(k) % self._pagesize]
            return self._preprocessor(x) if self._preprocessor is not None else x
        elif isinstance(k, slice):
            return [self[i] for i in range(len(self))[k.start if k.start else 0:k.stop if k.stop else len(self):k.step if k.step else 1]]  # expensive
        else:
            raise ValueError('invalid index type "%s"' % type(k))            

    def flush(self):
        self._pagecache = {}
        return self

        
    
class Union(Dataset):
    """vipy.dataset.Union() class
    
    Common class to manipulate groups of vipy.dataset.Dataset objects in parallel

    Usage:
    
        >>> vipy.dataset.Union(D1, D2, D3, id='union')
        >>> vipy.dataset.Union( (D1, D2, D3) )

    """
    
    def __init__(self, *args, **kwargs):
        datasets = args[0] if isinstance(args[0], (tuple, list)) else args
        assert all([isinstance(d, Dataset) for d in datasets])
        self._ds = datasets

        if 'index' in kwargs:
            self._idx = kwargs['index']
        else:
            self._idx = [(i,j) for j in range(max([len(d) for d in datasets])) for (i,d) in enumerate(datasets) if j<len(d)]  # zipped (dataset index, element index) tuples 
        self._id = kwargs['id'] if 'id' in kwargs else None

        self._preprocessor = None
        self._loader = None
        self._shuffler = kwargs['shuffler'] if 'shuffler' in kwargs else None
        
    def __iter__(self):
        for (i,j) in self._idx:
            yield self._ds[i][j]

    def __getitem__(self, k):
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(self._idx), "invalid index"
            (i,j) = self._idx[int(k)]            
            return self._ds[i][j]
        elif isinstance(k, slice):
            return [self._ds[i][j] for (i,j) in self._idx[k.start:k.stop:k.step]]
        else:
            raise ValueError('invalid index type "%s"' % type(k))

    def __repr__(self):
        fields = ['id=%s' % self.id(truncated=True, maxlen=64)] if self.id() else []
        fields += ['len=%d' % len(self)]
        fields += ['ids=%s' % [d.id(truncated=True, maxlen=32) for d in self._ds]]
        return str('<vipy.dataset.%s: %s>' % (self.__class__.__name__, ', '.join(fields)))
        
    def clone(self, shallow=False):
        """Return a copy of the dataset object"""
        D = super().clone(shallow=shallow)
        D._ds =  [d.clone(shallow=shallow) for d in D._ds]
        return D
    
    def datasets(self):
        """Return the dataset union elements, useful for generating unions of unions"""
        return list(self._ds)
    
    def raw(self):
        self._ds = [d.raw() for d in self._ds]  # in-place, clone() first 
        return self

    def preprocessor(self, f=None, remove=None):
        raise ValueError('unsupported')
    

    
class Collector(Dataset):
    """vipy.dataset.Collector() class
    
    Common class to manipulate datasets of vipy objects curated using Visym Collector

    ```python
    D = vipy.dataset.Collector([vipy.video.RandomScene(), vipy.video.RandomScene()], id='random_scene')
    with vipy.globals.parallel(2):
        D = D.map(lambda v: v.frame(0))
    list(D)
    ```

    Create dataset and export as a directory of json files 

    ```python
    D = vipy.dataset.Collector([vipy.video.RandomScene(), vipy.video.RandomScene()])
    D.tojsondir('/tmp/myjsondir')
    ```
    
    Create dataset from all json or pkl files recursively discovered in a directory and lazy loaded

    ```python
    D = vipy.dataset.Collector('/tmp/myjsondir')  # lazy loading
    ```

    Create dataset from a list of json or pkl files and lazy loaded

    ```python
    D = vipy.dataset.Collector(['/path/to/file1.json', '/path/to/file2.json'])  # lazy loading
    ```
    
    Args:
    
        - abspath [bool]: If true, load all lazy elements with absolute path
        - loader [lambda]: a callable loader that will process the object .  This is useful for custom deerialization or on demand transformations
        - lazy [bool]: If true, load all pkl or json files using the custom loader when accessed

    .. notes:: Be warned that using the jsondir constructor will load elements on demand, but there are some methods that require loading the entire dataset into memory, and will happily try to do so
    .. notes:: This class may be deprecated in the future
    
    """

    def __init__(self, objlist, id=None, loader=None, lazy=False):
        
        assert loader is None or callable(loader)

        self._id = id
        self._loader = self._default_loader if loader is None else loader  # may not be serializable if lambda is provided
        #self._istype_strict = True
        self._lazy_loader = lazy
        #self._abspath = abspath
        #self._shuffler = 'uniform'
        self._idx = list(range(len(objlist)))
        
        if isinstance(objlist, str) and (vipy.util.isjsonfile(objlist) or vipy.util.ispklfile(objlist) or vipy.util.ispklbz2(objlist)):
            self._objlist = vipy.util.load(objlist, abspath=abspath)
        elif isinstance(objlist, str) and os.path.isdir(objlist):
            self._objlist = vipy.util.findloadable(objlist) # recursive
            self._loader = lambda x,b=abspath:  vipy.util.load(x, abspath=b) if (vipy.util.ispkl(x) or vipy.util.isjsonfile(x) or vipy.util.ispklbz2(objlist)) else x
            self._istype_strict = False
            self._lazy_loader = True
        elif lazy and (isinstance(objlist, list) and all([(vipy.util.ispkl(x) or vipy.util.isjsonfile(x)) for x in objlist])):
            self._objlist = objlist 
            self._loader = lambda x,b=abspath:  vipy.util.load(x, abspath=b) if (vipy.util.ispkl(x) or vipy.util.isjsonfile(x)) else x            
            self._istype_strict = False
            self._lazy_loader = True
        else:
            self._objlist = objlist

        self._objlist = tolist(self._objlist)        
        assert len(self._objlist) > 0, "Empty dataset"

        if self._lazy_loader:
            try:
                self[0]
            except Exception as e:
                raise ValueError('Invalid dataset - Lazy load failed with error "%s"' % str(e))

    @staticmethod
    def _default_loader(x):
        return x
    
    def istype(self, validtype):
        """Return True if the all elements (or just the first element if strict=False) in the dataset are of type 'validtype'"""
        return all([any([isinstance(v,t) for t in tolist(validtype)]) for v in self]) if self._istype_strict else any([isinstance(self[0],t) for t in tolist(validtype)])
            
    def _isvipy(self):
        """Return True if all elements in the dataset are of type `vipy.video.Video` or `vipy.image.Image`"""        
        return self.istype([vipy.image.Image, vipy.video.Video])

    def _is_vipy_video(self):
        """Return True if all elements in the dataset are of type `vipy.video.Video`"""                
        return self.istype([vipy.video.Video])

    def _is_vipy_video_scene(self):
        """Return True if all elements in the dataset are of type `vipy.video.Scene`"""                        
        return self.istype([vipy.video.Scene])

    def _is_vipy_image_scene(self):
        """Return True if all elements in the dataset are of type `vipy.video.Scene`"""                        
        return self.istype([vipy.image.Scene])

    def json(self, encode=True):
        r = vipy.util.class_registry()
        d = {k:v for (k,v) in self.__dict__.items() if not k == '_loader'}
        d['_objlist'] = [(str(type(v)), v.json(encode=False)) if str(type(v)) in r else v for v in self._objlist]
        return json.dumps(d) if encode else d

    @classmethod
    def from_json(cls, s):
        r = vipy.util.class_registry()
        d = json.loads(s) if not isinstance(s, dict) else s  
        return cls(objlist=[r[x[0]](x[1]) if (isinstance(x, tuple) and x[0] in r) else x for x in d['_objlist']],
                   id=d['_id'],
                   abspath=d['_abspath'])                            
    
    def archive(self, tarfile, delprefix, mediadir='videos', format='json', castas=vipy.video.Scene, verbose=False, extrafiles=None, novideos=False, md5=True, tmpdir=None, inplace=False, bycategory=False, annotationdir='annotations'):
        """Create a archive file for this dataset.  This will be archived as:

           /path/to/tarfile.{tar.gz|.tgz|.bz2}
              tarfilename
                 tarfilename.{json|pkl}
                 mediadir/
                     video.mp4
                 extras1.ext
                 extras2.ext
        
            Args:
                tarfile: /path/to/tarfilename.tar.gz
                delprefix:  the absolute file path contained in the media filenames to be removed.  If a video has a delprefix='/a/b' then videos with path /a/b/c/d.mp4' -> 'c/d.mp4', and {JSON|PKL} will be saved with relative paths to mediadir.  This may be a list of delprefixes.
                mediadir:  the subdirectory name of the media to be contained in the archive.  Usually "videos".             
                extrafiles: list of tuples or singletons [(abspath, filename_in_archive_relative_to_root), 'file_in_root_and_in_pwd', ...], 
                novideos [bool]:  generate a tarball without linking videos, just annotations
                md5 [bool]:  If True, generate the MD5 hash of the tarball using the system "md5sum", or if md5='vipy' use a slower python only md5 hash 
                castas [class]:  This should be a vipy class that the vipy objects should be cast to prior to archive.  This is useful for converting priveledged superclasses to a base class prior to export.
                tmpdir:  The path to the temporary directory for construting this dataset.  Defaults to system temp.  This directory will be emptied prior to archive.
                inplace [bool]:  If true, modify the dataset in place to prepare it for archive, else make a copy
                bycategory [bool]: If true, save the annotations in an annotations/ directory by category
                annotationdir [str]: The subdirectory name of annotations to be contained in the archive if bycategory=True.  Usually "annotations" or "json".

            Example:  

              - Input files contain /path/to/oldvideos/category/video.mp4
              - Output will contain relative paths videos/category/video.mp4

        ```python
        d.archive('out.tar.gz', delprefix='/path/to/oldvideos', mediadir='videos')
        ```
        
            Returns:

                The absolute path to the tarball 
        """
        assert self._isvipy(), "Source dataset must contain vipy objects for staging"
        assert all([os.path.isabs(v.filename()) for v in self]), "Input dataset must have only absolute media paths"
        assert len([v for v in self if any([d in v.filename() for d in tolist(delprefix)])]) == len(self), "all media objects must have a provided delprefix for relative path construction"
        assert vipy.util.istgz(tarfile) or vipy.util.istarbz2(tarfile) or vipy.util.istar(tarfile), "Allowable extensions are .tar.gz, .tgz, .bz2 or .tar"
        assert shutil.which('tar') is not None, "tar not found on path"        
        
        D = self.clone() if not inplace else self   # large memory footprint if inplace=False
        tmpdir = tempdir() if tmpdir is None else remkdir(tmpdir, flush=True)
        stagedir = remkdir(os.path.join(tmpdir, filefull(filetail(tarfile))))
        print('[vipy.dataset]: creating staging directory "%s"' % stagedir)
        delprefix = [[d for d in tolist(delprefix) if d in v.filename()][0] for v in self]  # select the delprefix per video
        D._objlist = [v.filename(v.filename().replace(os.path.normpath(p), os.path.normpath(os.path.join(stagedir, mediadir))), symlink=not novideos) for (p,v) in zip(delprefix, D.list())]

        # Save annotations:  Split large datasets into annotations grouped by category to help speed up loading         
        if bycategory:
            for (c,V) in vipy.util.groupbyasdict(list(D), lambda v: v.category()).items():
                Dataset(V, id=c).save(os.path.join(stagedir, annotationdir, '%s.%s' % (c, format)), relpath=True, nourl=True, sanitize=True, castas=castas, significant_digits=2, noemail=True, flush=True)
        else:
            pklfile = os.path.join(stagedir, '%s.%s' % (filetail(filefull(tarfile)), format))
            D.save(pklfile, relpath=True, nourl=True, sanitize=True, castas=castas, significant_digits=2, noemail=True, flush=True)
    
        # Copy extras (symlinked) to staging directory
        if extrafiles is not None:
            # extrafiles = [("/abs/path/in/filesystem.ext", "rel/path/in/archive.ext"), ... ]
            assert all([((isinstance(e, tuple) or isinstance(e, list)) and len(e) == 2) or isinstance(e, str) for e in extrafiles])
            extrafiles = [e if (isinstance(e, tuple) or isinstance(e, list)) else (e,e) for e in extrafiles]  # tuple-ify files in pwd() and should be put in the tarball root
            for (e, a) in tolist(extrafiles):
                assert os.path.exists(os.path.abspath(e)), "Invalid extras file '%s' - file not found" % e
                remkdir(filepath(os.path.join(stagedir, filetail(e) if a is None else a)))    # make directory in stagedir for symlink
                os.symlink(os.path.abspath(e), os.path.join(stagedir, filetail(e) if a is None else a))

        # System command to run tar
        cmd = ('tar %scvf %s -C %s --dereference %s %s' % ('j' if vipy.util.istarbz2(tarfile) else ('z' if vipy.util.istgz(tarfile) else ''), 
                                                           tarfile,
                                                           filepath(stagedir),
                                                           filetail(stagedir),
                                                           ' > /dev/null' if not verbose else ''))

        print('[vipy.dataset]: executing "%s"' % cmd)        
        os.system(cmd)  # too slow to use python "tarfile" package
        print('[vipy.dataset]: deleting staging directory "%s"' % stagedir)        
        shutil.rmtree(stagedir)

        if md5:
            if shutil.which('md5sum') is not None:
                cmd = 'md5sum %s' % tarfile
                print('[vipy.dataset]: executing "%s"' % cmd)        
                os.system(cmd)  # too slow to use python "vipy.downloader.generate_md5(tarball)" for huge datasets
            else:
                print('[vipy.dataset]: %s, MD5=%s' % (tarfile, vipy.downloader.generate_md5(tarfile)))  # too slow for large datasets, but does not require md5sum on path
        return tarfile
        
    def save(self, outfile, nourl=False, castas=None, relpath=False, sanitize=True, strict=True, significant_digits=2, noemail=True, flush=True, bycategory=False):
        """Save the dataset to the provided output filename stored as pkl or json
        
        Args:
            outfile: [str]: The /path/to/out.pkl or /path/to/out.json
            nourl: [bool]: If true, remove all URLs from the media (if present)
            castas: [type]:  Cast all media to the provided type.  This is useful for downcasting to `vipy.video.Scene` from superclasses
            relpath: [bool]: If true, define all file paths in objects relative to the /path/to in /path/to/out.json
            sanitize: [bool]:  If trye, call sanitize() on all objects to remove all private attributes with prepended '__' 
            strict: [bool]: Unused
            significant_digits: [int]: Assign the requested number of significant digits to all bounding boxes in all tracks.  This requires dataset of `vipy.video.Scene`
            noemail: [bool]: If true, scrub the attributes for emails and replace with a hash
            flush: [bool]:  If true, flush the object buffers prior to save
            bycategory [bool[: If trye, then save the dataset to the provided output filename pattern outfile='/path/to/annotations/*.json' where the wildcard is replaced with the category name

        Returns:        
            This dataset that is quivalent to vipy.dataset.Collector('/path/to/outfile.json')
        """
        n = len([v for v in self if v is None])
        if n > 0:
            print('[vipy.dataset]: removing %d invalid elements' % n)
        objlist = [v for v in self if v is not None]  
        if relpath or nourl or sanitize or flush or noemail or (significant_digits is not None):
            assert self._isvipy(), "Invalid input"
        if relpath:
            print('[vipy.dataset]: setting relative paths')
            objlist = [v.relpath(start=filepath(outfile)) if os.path.isabs(v.filename()) else v for v in objlist]
        if nourl: 
            print('[vipy.dataset]: removing URLs')
            objlist = [v.nourl() for v in objlist]           
        if sanitize:
            print('[vipy.dataset]: sanitizing attributes')                        
            objlist = [v.sanitize() for v in objlist]  # removes all attributes with '__' keys
        if castas is not None:
            assert hasattr(castas, 'cast'), "Invalid cast"
            print('[vipy.dataset]: casting as "%s"' % (str(castas)))
            objlist = [castas.cast(v) for v in objlist]                     
        if significant_digits is not None:
            assert self._is_vipy_video_scene()
            assert isinstance(significant_digits, int) and significant_digits >= 1, "Invalid input"
            objlist = [o.trackmap(lambda t: t.significant_digits(significant_digits)) if o is not None else o for o in objlist]
        if noemail:
            print('[vipy.dataset]: removing emails')            
            for o in objlist:
                for (k,v) in o.attributes.items():
                    if isinstance(v, str) and is_email_address(v):
                        o.attributes[k] = hashlib.sha1(v.encode("UTF-8")).hexdigest()[0:10]
        if flush:
            objlist = [o.flush() for o in objlist]  

        if bycategory:
            for (c,V) in vipy.util.groupbyasdict(list(self), lambda v: v.category()).items():
                jsonfile = outfile.replace('*', c)  # outfile="/path/to/annotations/*.json"
                d = Dataset(V, id=c).save(jsonfile, relpath=relpath, nourl=nourl, sanitize=sanitize, castas=castas, significant_digits=significant_digits, noemail=noemail, flush=flush, bycategory=False)
                print('[vipy.dataset]: Saving %s by category to "%s"' % (str(d), jsonfile))                
        else:
            print('[vipy.dataset]: Saving %s to "%s"' % (str(self), outfile))
            vipy.util.save(objlist, outfile)
        return self

    def classlist(self):
        """Return a sorted list of categories in the dataset"""
        return sorted(list(set([v.category() for v in self])))

    def classes(self):
        """Alias for classlist"""
        return self.classlist()
    def categories(self):
        """Alias for classlist"""
        return self.classlist()
    def num_classes(self):
        """Return the number of unique categories in this dataset"""
        return len(self.classlist())
    def num_labels(self):
        """Alias for num_classes"""
        return self.num_classes()
    def num_categories(self):
        """Alias for num_classes"""
        return self.num_classes()
    
    
    def class_to_index(self):
        """Return a dictionary mapping the unique classes to an integer index.  This is useful for defining a softmax index ordering for categorization"""
        return {v:k for (k,v) in enumerate(self.classlist())}

    def index_to_class(self):
        """Return a dictionary mapping an integer index to the unique class names.  This is the inverse of class_to_index, swapping keys and values"""
        return {v:k for (k,v) in self.class_to_index().items()}

    def label_to_index(self):
        """Alias for class_to_index"""
        return self.class_to_index()

    def powerset(self):
        return list(sorted(set([tuple(sorted(list(a))) for v in self for a in v.activitylabel() if len(a) > 0])))        

    def powerset_to_index(self):        
        return {c:k for (k,c) in enumerate(self.powerset())}

    def merge(self, outdir):
        """Merge a dataset union into a single subdirectory with symlinked media ready to be archived.

        ```python
        D1 = vipy.dataset.Collector('/path1/dataset.json')
        D2 = vipy.dataset.Collector('/path2/dataset.json')
        D3 = D1.union(D2).merge(outdir='/path3')
        ```

        Media in D1 are in /path1, media in D2 are in /path2, media in D3 are all symlinked to /path3.
        We can now create a tarball for D3 with all of the media files in the same relative path.
        """
        
        outdir = vipy.util.remkdir(os.path.abspath(os.path.normpath(outdir)))
        return self.clone().localmap(lambda v: v.filename(os.path.join(outdir, filetail(v.filename())), copy=False, symlink=True))

    def augment(self, f, n_augmentations):
        assert n_augmentations >= 1
        self._objlist = [f(v.clone()) for v in self for k in range(n_augmentations)]  # This will remove the originals
        return self

    def jsondir(self, outdir=None, verbose=True, rekey=False, bycategory=False, byfilename=False, abspath=True):
        """Export all objects to a directory of JSON files.
    
           Usage:

        ```python
        D = vipy.dataset.Collector(...).jsondir('/path/to/jsondir')
        D = vipy.util.load('/path/to/jsondir')   # recursively discover and lazy load all json files 
        ```

           Args:
               outdir [str]:  The root directory to store the JSON files
               verbose [bool]: If True, print the save progress
               rekey [bool] If False, use the instance ID of the vipy object as the filename for the JSON file, otherwise assign a new UUID_dataset-index
               bycategory [bool]: If True, use the JSON structure '$OUTDIR/$CATEGORY/$INSTANCEID.json'
               byfilename [bool]: If True, use the JSON structure '$FILENAME.json' where $FILENAME is the underlying media filename of the vipy object
               abspath [bool]: If true, store absolute paths to media in JSON.  If false, store relative paths to media from JSON directory

           Returns:
               outdir: The directory containing the JSON files.
        """
        assert outdir is not None or byfilename 
        assert not byfilename and bycategory

        if outdir is not None:
            vipy.util.remkdir(outdir) 
        if bycategory:
            tojsonfile = lambda v,k: os.path.join(outdir, v.category(), ('%s.json' % v.instanceid()) if not rekey else ('%s_%d.json' % (uuid.uuid4().hex, k)))
        elif byfilename:
            tojsonfile = lambda v,k: vipy.util.toextension(v.filename(), '.json')
        else:
            tojsonfile = lambda v,k: os.path.join(outdir, ('%s.json' % v.instanceid()) if not rekey else '%s_%d.json' % (uuid.uuid4().hex, k))
        
        for (k,v) in enumerate(self):            
            f = vipy.util.save(v.clone().relpath(start=filepath(tojsonfile(v,k))) if not abspath else v.clone().abspath(), tojsonfile(v,k))
            if verbose:
                print('[vipy.dataset.Collector][%d/%d]: %s' % (k, len(self), f))
        return outdir

    def tojsondir(self, outdir=None, verbose=True, rekey=False, bycategory=False, byfilename=False, abspath=True):
        """Alias for `vipy.dataset.Collector.jsondir`"""
        return self.jsondir(outdir, verbose=verbose, rekey=rekey, bycategory=bycategory, byfilename=byfilename, abspath=abspath)
    
    def take_per_category(self, n):
        """Random;y take n elements per category and return a shallow cloned dataset"""
        D = self.clone(shallow=True)
        d_category_to_objlist = vipy.util.groupbyasdict(self._objlist, lambda x: x.category())
        D._objlist = [v for c in self.categories() for v in Dataset(d_category_to_objlist[c]).take(n)]
        return D

    def shuffler(self, method=None, uniform=None, pairwise=None):
        """Specify a shuffler protocol.  
        
           >>> D.shuffler('uniform')
           >>> D.shuffer(uniform=True)
           >>> D.shuffle()

           Args:
             uniform [bool]: shuffle element uniformly at random
             pairwise [bool]:  elements are assumed to be pairwise similarities, such that the category() method returns an id for each positive pair.  Shuffle keeping positive pairs as minibatch neighbors.        

           Returns: self if a new shuffler is requested, otherwise return a lambda function which shuffles a list. This lambda function is not meant to be used directly, rather exercised by shuffle
        """
        if method:
            assert method in ['uniform', 'pairwise'], "unknown shuffler '%s'" % method
            self._shuffler = method
        elif pairwise:
            self._shuffler = 'pairwise'
        elif uniform:
            self._shuffler = 'uniform'
        elif self._shuffler == 'uniform':
            return lambda y: sorted(y, key=lambda x: random.random())
        elif self._shuffler == 'pairwise':
            return lambda y: vipy.util.flatlist(sorted(vipy.util.chunklistbysize(sorted(y, key=lambda x: x.category()), 2), key=lambda x: random.random()))
        return self
    
    def shuffle(self):
        """Randomly permute elements in this dataset according to a shuffler protocol set with shuffler()"""
        self._objlist = self.shuffler()(self._objlist)  # in-place
        return self
    
    def _split_by_videoid(self, trainfraction=0.9, valfraction=0.1, testfraction=0, seed=None):
        """Split the dataset by category by fraction so that video IDs are never in the same set"""
        assert trainfraction >=0 and trainfraction <= 1
        assert valfraction >=0 and valfraction <= 1
        assert testfraction >=0 and testfraction <= 1
        assert trainfraction + valfraction + testfraction == 1.0
        np.random.seed(seed)  # deterministic
        
        # Video ID assignment
        A = self.list()
        videoid = list(set([a.videoid() for a in A]))
        np.random.shuffle(videoid)
        (testid, valid, trainid) = vipy.util.dividelist(videoid, (testfraction, valfraction, trainfraction))        
        (testid, valid, trainid) = (set(testid), set(valid), set(trainid))
        d = groupbyasdict(A, lambda a: 'testset' if a.videoid() in testid else 'valset' if a.videoid() in valid else 'trainset')
        (trainset, testset, valset) = (d['trainset'] if 'trainset' in d else [], 
                                       d['testset'] if 'testset' in d else [], 
                                       d['valset'] if 'valset' in d else [])

        #print('[vipy.dataset]: trainset=%d (%1.2f)' % (len(trainset), trainfraction))
        #print('[vipy.dataset]: valset=%d (%1.2f)' % (len(valset), valfraction))
        #print('[vipy.dataset]: testset=%d (%1.2f)' % (len(testset), testfraction))
        np.random.seed()  # re-initialize seed

        return (Dataset(trainset, id='trainset'), Dataset(valset, id='valset'), Dataset(testset, id='testset') if len(testset)>0 else None)

    def tocsv(self, csvfile=None):
        csv = [v.csv() for v in self.list]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])


    def map(self, f_map, model=None, dst=None, id=None, strict=False, ascompleted=True, ordered=False):        
        """Distributed map.

        To perform this in parallel across four processes:

        ```python
        D = vipy.dataset.Collector(...)
        with vipy.globals.parallel(4):
            D.map(lambda v: ...)
        ```

        Args:
            f_map: [lambda] The lambda function to apply in parallel to all elements in the dataset.  This must return a JSON serializable object
            model: [torch.nn.Module] The model to scatter to all workers
            dst: [str] The ID to give to the resulting dataset
            id: [str] The ID to give to the resulting dataset (parameter alias for dst)
            strict: [bool] If true, raise exception on map failures, otherwise the map will return None for failed elements
            ascompleted: [bool] If true, return elements as they complete
            ordered: [bool] If true, preserve the order of objects in dataset as returned from distributed processing
        
        Returns:
            A `vipy.dataset.Collector` containing the elements f_map(v).  This operation is order preserving if ordered=True.

        .. note:: 
            - This dataset must contain vipy objects of types defined in `vipy.util.class_registry` or JSON serializable objects
            - Serialization of large datasets can take a while, kick it off to a distributed dask scheduler and go get lunch
            - This method uses dask distributed and `vipy.batch.Batch` operations
            - All vipy objects are JSON serialized prior to parallel map to avoid reference cycle garbage collection which can introduce instabilities
            - Due to chunking, all error handling is caught by this method.  Use `vipy.batch.Batch` to leverage dask distributed futures error handling.
            - Operations must be chunked and serialized because each dask task comes with overhead, and lots of small tasks violates best practices
            - Serialized results are deserialized by the client and returned a a new dataset
        """
        assert callable(f_map)    
        from vipy.batch import Batch   # requires pip install vipy[all]

        # Distributed map using vipy.batch
        f_serialize = lambda v,d=vipy.util.class_registry(): (str(type(v)), v.json()) if str(type(v)) in d else (None, pickle.dumps(v))  # fallback on PKL dumps/loads
        f_deserialize = lambda x,d=vipy.util.class_registry(): d[x[0]](x[1])  # with closure capture
        f_catcher = lambda f, *args, **kwargs: vipy.util.loudcatcher(f, '[vipy.dataset.Collector.map]: ', *args, **kwargs)  # catch exceptions when executing lambda, print errors and return (True, result) or (False, exception)
        f_loader = self._loader if self._loader is not None else lambda x: x
        S = [f_serialize(v) for v in self._objlist]  # local serialization

        B = Batch(vipy.util.chunklist(S, 128), strict=strict, as_completed=ascompleted, warnme=False, minscatter=128, ordered=ordered)
        if model is None:
            f = lambda x, f_loader=f_loader, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher: f_serializer(f_catcher(f_map, f_loader(f_deserializer(x))))  # with closure capture
            S = B.map(lambda X,f=f: [f(x) for x in X]).result()  # chunked, with caught exceptions, may return empty list
        else:
            f = lambda net, x, f_loader=f_loader, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher: f_serializer(f_catcher(f_map, net, f_loader(f_deserializer(x))))  # with closure capture
            S = B.scattermap((lambda net, X, f=f: [f(net, x) for x in X]), model).result()  # chunked, scattered, caught exceptions
        if not isinstance(S, list) or any([not isinstance(s, list) for s in S]):
            raise ValueError('Distributed processing error - Batch returned: %s' % (str(S)))
        V = [f_deserialize(x) for s in S for x in s]  # Local deserialization and chunk flattening
        (good, bad) = ([r for (b,r) in V if b], [r for (b,r) in V if not b])  # catcher returns (True, result) or (False, exception string)
        if len(bad) > 0:
            print('[vipy.dataset.Collector.map]: Exceptions in map distributed processing:\n%s' % str(bad))
            print('[vipy.dataset.Collector.map]: %d/%d items failed' % (len(bad), len(self)))
        return Dataset(good, id=dst if dst is not None else id)
    
    def synonym(self, synonymdict):
        """Convert all categories in the dataset using the provided synonym dictionary mapping"""
        assert isinstance(synonymdict, dict)
        
        if self._is_vipy_video_scene():
            return self.localmap(lambda v: v.trackmap(lambda t: t.categoryif(synonymdict)).activitymap(lambda a: a.categoryif(synonymdict)))
        elif self._is_vipy_image_scene():
            return self.localmap(lambda v: v.objectmap(lambda o: o.categoryif(synonymdict)))
        return self

    def histogram(self, outfile=None, fontsize=6, category_to_barcolor=None, category_to_xlabel=None, ylabel='Instances'):
        assert category_to_barcolor is None or all([c in category_to_barcolor for c in self.categories()])
        assert category_to_xlabel is None or callable(category_to_xlabel) or all([c in category_to_xlabel for c in self.categories()])
        f_category_to_xlabel = category_to_xlabel if callable(category_to_xlabel) else ((lambda c: category_to_xlabel[c]) if category_to_xlabel is not None else (lambda c: c))
        
        d = self.countby(lambda v: v.category())
        if outfile is not None:
            (categories, freq) = zip(*reversed(sorted(list(d.items()), key=lambda x: x[1])))  # decreasing frequency
            barcolors = ['blue' if category_to_barcolor is None else category_to_barcolor[c] for c in categories]
            xlabels = [f_category_to_xlabel(c) for c in categories]
            vipy.metrics.histogram(freq, xlabels, barcolors=barcolors, outfile=outfile, ylabel=ylabel, fontsize=fontsize)
        return d

    def percentage(self):
        """Fraction of dataset for each label"""
        d = self.count(lambda v: v.category())
        n = sum(d.values())
        return {k:v/float(n) for (k,v) in d.items()}


    def multilabel_inverse_frequency_weight(self):
        """Return an inverse frequency weight for multilabel activities, where label counts are the fractional label likelihood within a clip"""
        assert self._is_vipy_video()

        def _multilabel_inverse_frequency_weight(v):
            lbl_likelihood = {}
            if len(v.activities()) > 0:
                (ef, sf) = (max([a.endframe() for a in v.activitylist()]), min([a.startframe() for a in v.activitylist()]))  # clip length 
                lbl_list = [a for A in v.activitylabel(sf, ef) for a in set(A)]  # list of all labels within clip (labels are unique in each frame)
                lbl_frequency = vipy.util.countby(lbl_list, lambda x: x)  # frequency of each label within clip
                lbl_weight = {k:v/float(len(lbl_list)) for (k,v) in lbl_frequency.items()}  # multi-label likelihood within clip, normalized frequency sums to one 
                for (k,w) in lbl_weight.items():
                    if k not in lbl_likelihood:
                        lbl_likelihood[k] = 0
                    lbl_likelihood[k] += w
            return lbl_likelihood
                    
        lbl_likelihood  = {}
        for d in self.map(lambda v: _multilabel_inverse_frequency_weight(v)):  # parallelizable
            for (k,v) in d.items():
                if k not in lbl_likelihood:
                    lbl_likelihood[k] = 0
                lbl_likelihood[k] += v

        # Inverse frequency weight on label likelihood per clip
        d = {k:1.0/max(v,1) for (k,v) in lbl_likelihood.items()}
        n = sum(d.values())  
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def inverse_frequency_weight(self):
        """Return inverse frequency weight for categories in dataset.  Useful for unbalanced class weighting during training"""
        d = {k:1.0/max(v,1) for (k,v) in self.count().items()}
        n = sum(d.values())
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def duration_in_frames(self, outfile=None):
        d = {k:np.mean([v[1] for v in v]) for (k,v) in groupbyasdict([(a.category(), len(a)) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            vipy.metrics.histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (frames)', fontsize=6)            
        return d

    def duration_in_seconds(self, outfile=None, fontsize=6, max_duration=None):
        """Duration of activities"""
        d = {k:np.mean([v[1] for v in v]) for (k,v) in groupbyasdict([(a.category(), len(a)/v.framerate()) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            max_duration = max(d.values()) if max_duration is None else max_duration
            vipy.metrics.histogram([min(x, max_duration) for x in d.values()], d.keys(), outfile=outfile, ylabel='Duration (seconds)', fontsize=fontsize)            
        return d

    def video_duration_in_seconds(self, outfile=None, fontsize=6, max_duration=None):
        """Duration of activities"""
        d = {k:np.mean([d for (c,d) in D]) for (k,D) in groupbyasdict([(v.category(), v.duration()) for v in self.list()], lambda x: x[0]).items()}
        if outfile is not None:
            max_duration = max(d.values()) if max_duration is None else max_duration
            vipy.metrics.histogram([min(x, max_duration) for x in d.values()], d.keys(), outfile=outfile, ylabel='Duration (seconds)', fontsize=fontsize)            
        return d
    
    def framerate(self, outfile=None):
        d = vipy.util.countby([int(round(v.framerate())) for v in self.list()], lambda x: x)
        if outfile is not None:
            vipy.metrics.pie(d.values(), ['%d fps' % k for k in d.keys()], explode=None, outfile=outfile,  shadow=False)
        return d
        
        
    def density(self, outfile=None, max=None):
        """Compute the frequency that each video ID is represented.  This counts how many activities are in a video, truncated at max"""
        d = [len(v) if (max is None or len(v)<= max) else max for (k,v) in groupbyasdict(self.list(), lambda v: v.videoid()).items()]
        d = {k:v for (k,v) in sorted(vipy.util.countby(d, lambda x: x).items(), key=lambda x: x[1], reverse=True)}
        if outfile is not None:
            vipy.metrics.histogram(d.values(), d.keys(), outfile=outfile, ylabel='Frequency', xlabel='Activities per video', fontsize=6, xrot=None)            
        return d

    def boxsize(self, outfile=None, category_to_color=None, categories=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks]) if categories is None else categories

        d = {}        
        for c in object_categories:
            xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category().lower() == c.lower()) and (t.meanshape() is not None))]
            d[c] = xcyc
        
        if outfile is not None:            
            plt.clf()
            plt.figure()
            plt.grid(True)
            for c in object_categories:
                xcyc = d[c]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.scatter(xc, yc, c=category_to_color[c] if category_to_color is not None else 'blue', label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 1000, 0, 1000])                
            plt.legend()
            plt.gca().set_axisbelow(True)        
            plt.savefig(outfile)
        return d

    def boxsize_by_category(self, outfile=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks])
        
        # Mean track size per video category
        d_category_to_xy = {k:np.mean([t.meanshape() for v in vlist for t in v.tracklist()], axis=0) for (k,vlist) in groupbyasdict(self.list(), lambda v: v.category()).items()}

        if outfile is not None:
            plt.clf()
            plt.figure()
            plt.grid(True)
            colors = colorlist()            
            d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(d_category_to_xy.keys())}
            for c in d_category_to_xy.keys():
                (xc, yc) = d_category_to_xy[c]
                plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 600, 0, 600])                
            plt.gca().set_axisbelow(True)        
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.savefig(outfile, bbox_extra_artists=(lgd,), bbox_inches='tight')
        return d_category_to_xy

    def boxsize_histogram(self, outfile=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks])

        
        # 2D histogram of object box sizes
        for c in object_categories:
            xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category() == c) and (t.meanshape() is not None))]
            d[c] = xcyc

        if outfile is not None:
            for c in object_categories:            
                xcyc = d[c]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.clf()
                    plt.figure()
                    plt.hist2d(xc, yc, bins=10)
                    plt.xlabel('Bounding box (width)')
                    plt.ylabel('Bounding box (height)')                    
                    plt.savefig(outfile % c)
        return d

    def to_torch(self, f_video_to_tensor):
        """Return a torch dataset that will apply the lambda function f_video_to_tensor to each element in the dataset on demand"""
        import vipy.torch
        return vipy.torch.TorchDataset(f_video_to_tensor, self)

    def to_torch_tensordir(self, f_video_to_tensor, outdir, n_augmentations=20, sleep=None):
        """Return a TorchTensordir dataset that will load a pkl.bz2 file that contains one of n_augmentations (tensor, label) pairs.
        
        This is useful for fast loading of datasets that contain many videos.

        """
        import vipy.torch    # lazy import, requires vipy[all] 
        from vipy.batch import Batch   # requires pip install vipy[all]

        assert self._is_vipy_video_scene()
        outdir = vipy.util.remkdir(outdir)
        self.map(lambda v, f=f_video_to_tensor, outdir=outdir, n_augmentations=n_augmentations: vipy.util.bz2pkl(os.path.join(outdir, '%s.pkl.bz2' % v.instanceid()), [f(v.print(sleep=sleep).clone()) for k in range(0, n_augmentations)]))
        return vipy.torch.Tensordir(outdir)

    def annotate(self, outdir, mindim=512):
        f = lambda v, outdir=outdir, mindim=mindim: v.mindim(mindim).annotate(outfile=os.path.join(outdir, '%s.mp4' % v.videoid())).print()
        return self.map(f, dst='annotate')

    def tohtml(self, outfile, mindim=512, title='Visualization', fraction=1.0, display=False, clip=True, activities=True, category=True):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
        assert ishtml(outfile), "Output file must be .html"
        assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"
        
        import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
        import vipy.batch  # requires pip install vipy[all]

        dataset = self.list()
        assert all([isinstance(v, vipy.video.Video) for v in dataset])
        dataset = [dataset[int(k)] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
        #dataset = [v for v in dataset if all([len(a) < 15*v.framerate() for a in v.activitylist()])]  # remove extremely long videos

        quicklist = vipy.batch.Batch(dataset, strict=False, as_completed=True, minscatter=1).map(lambda v: (v.load().quicklook(), v.flush().print())).result()
        quicklist = [x for x in quicklist if x is not None]  # remove errors
        quicklooks = [imq for (imq, v) in quicklist]  # keep original video for HTML display purposes
        provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)


    def video_montage(self, outfile, gridrows, gridcols, mindim=64, bycategory=False, category=None, annotate=True, trackcrop=False, transpose=False, max_duration=None, framerate=30, fontsize=8):
        """30x50 activity montage, each 64x64 elements.

        Args:
            outfile: [str] The name of the outfile for the video.  Must have a valid video extension. 
            gridrows: [int, None]  The number of rows to include in the montage.  If None, infer from other args
            gridcols: [int] The number of columns in the montage
            mindim: [int] The square size of each video in the montage
            bycategory: [bool]  Make the video such that each row is a category 
            category: [str, list] Make the video so that every element is of category.  May be a list of more than one categories
            annotate: [bool] If true, include boxes and captions for objects and activities
            trackcrop: [bool] If true, center the video elements on the tracks with dilation factor 1.5
            transpose: [bool] If true, organize categories columnwise, but still return a montage of size (gridrows, gridcols)
            max_duration: [float] If not None, then set a maximum duration in seconds for elements in the video.  If None, then the max duration is the duration of the longest element.

        Returns:
            A clone of the dataset containing the selected videos for the montage, ordered rowwise in the montage

        .. notes::  
            - If a category does not contain the required number of elements for bycategory, it is removed prior to visualization
            - Elements are looped if they exit prior to the end of the longest video (or max_duration)
        """
        assert self._is_vipy_video()
        assert vipy.util.isvideo(outfile)
        assert gridrows is None or (isinstance(gridrows, int) and gridrows >= 1)
        assert gridcols is None or (isinstance(gridcols, int) and gridcols >= 1)
        assert isinstance(mindim, int) and mindim >= 1
        assert category is None or isinstance(category, str)

        D = self.clone()
        if bycategory:
            (num_categories, num_elements) = (gridrows, gridcols) if not transpose else (gridcols, gridrows)
            assert num_elements is not None
            requested_categories = sorted(D.classlist()) if (num_categories is None) else sorted(D.classlist())[0:num_categories]             
            categories = [c for c in requested_categories if D.count()[c] >= num_elements]  # filter those categories that do not have enough
            if set(categories) != set(requested_categories):
                warnings.warn('[vipy.dataset.video_montage]: removing "%s" without at least %d examples' % (str(set(requested_categories).difference(set(categories))), num_elements))
            vidlist = sorted(D.filter(lambda v: v.category() in categories).take_per_category(num_elements).tolist(), key=lambda v: v.category())
            vidlist = vidlist if not transpose else [vidlist[k] for k in np.array(range(0, len(vidlist))).reshape( (len(categories), num_elements) ).transpose().flatten().tolist()] 
            (gridrows, gridcols) = (len(categories), num_elements) if not transpose else (num_elements, len(categories))
            assert len(vidlist) == gridrows*gridcols

        elif category is not None:
            vidlist = D.filter(lambda v: v.category() in vipy.util.tolist(category)).take(gridrows*gridcols, canload=True).tolist()            
        elif len(D) != gridrows*gridcols:
            vidlist = D.take(gridrows*gridcols, canload=True).tolist()
        else:
            vidlist = D.tolist()

        vidlist = [v.framerate(framerate) for v in vidlist]  # resample to common framerate (this may result in jittery tracks
        montage = Dataset(vidlist, id='video_montage').clone()  # for output
        vidlist = [v.trackcrop(dilate=1.5, maxsquare=True) if (v.trackbox() is not None) else v for v in vidlist] if trackcrop else vidlist  # may be None, if so return the video
        vidlist = [v.mindim(mindim) for v in vidlist]  # before annotate for common font size
        vidlist = [vipy.video.Video.cast(v) for v in vidlist] if not annotate else [v.annotate(verbose=False, fontsize=fontsize) for v in vidlist]  # pre-annotate
            
        vipy.visualize.videomontage(vidlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols, framerate=framerate, max_duration=max_duration).saveas(outfile)
        return montage        
    
    def dedupe(self, key):
        assert callable(key)
        self._ds = list({key(v):v for v in self}.values())
        return self
    
    def union(self, other, key=None):
        assert isinstance(other, Dataset), "invalid input"
        if len(other) > 0:
            try:
                if other._loader is not None:
                    other._loader(self._ds[0])
                if self._loader is not None:
                    self._loader(other._ds[0])
                self._ds = self._ds + other._ds  # compatible loaders
            except:
                self._ds = self.list() + other.list()  # incompatible loaders
                self._loader = None
        return self.dedupe(key) if key is not None else self
    
    def difference(self, other, key):
        assert isinstance(other, Dataset), "invalid input"
        idset = set([key(v) for v in self]).difference([key(v) for v in other])   # in A but not in B
        self._ds = [v for v in self if key(v) in idset]
        return self
        
    def has(self, val, key):
        assert callable(key)
        return any([key(obj) == val for obj in self])

    def replace(self, other, key):
        """Replace elements in self with other with equality detemrined by the key lambda function"""
        assert isinstance(other, Dataset), "invalid input"
        assert callable(key)
        d = {key(v):v for v in other}
        self._ds = [v if key(v) not in d else d[key(v)] for v in self]
        return self
    
    def valid(self):
        return self.filter(lambda v: v is not None)

    def takefilter(self, f, n):
        """Apply the lambda function f and return n elements in a list where the filter lambda returns true
        
        Args:
            f: [lambda] If f(x) returns true, then keep
            n: [int >= 0] The number of elements to take
        
        Returns:
            [n=0] Returns empty list
            [n=1] Returns singleton element
            [n>1] Returns list of elements of at most n such that each element f(x) is True            
        """
        objlist = [obj for obj in self if f(obj)]
        return [] if (len(objlist) == 0 or n == 0) else (objlist[0] if n==1 else objlist[0:n])
    
    def flatmap(self, f):
        self._ds = [x for v in self for x in f(v)]
        return self

    def repeat(self, n):
        """Clone the elements in this dataset and repeat n times.  The length of the new dataset will be (n+1)*len(self)"""
        D = self.clone()
        for k in range(n):
            D._ds.extend(self.clone()._ds)
        return D

    def frequency(self, f):
        """synonym for self.count"""
        return self.count(f)
    
    def flatten(self):
        """Convert dataset stored as a list of lists into a flat list"""
        self._ds = [o for objlist in self._ds for o in vipy.util.tolist(objlist)]
        return self

    def zip(self, other):
        """Zip two datasets.  Equivalent to zip(self, other).

        ```python
        for (d1,d2) in D1.zip(D2, sortkey=lambda v: v.instanceid()):
            pass
        
        for (d1, d2) in zip(D1, D2):
            pass
        ```

        Args:
            other: [`vipy.dataset.Dataset`] 
            sortkey: [lambda] sort both datasets using the provided sortkey lambda.
        
        Returns:
            Generator for the tuple sequence ( (self[0], other[0]), (self[1], other[1]), ... )
        """ 
        assert isinstance(other, Dataset)
        assert len(self) == len(other)

        for (vi, vj) in zip(self, other):
            yield (vi, vj)
    
