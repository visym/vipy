import os
import numpy as np
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml
from vipy.util import listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv, to_iterable
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
        - index [list]: If provided, use this as the initial index into the dataset.  This is useful for preprocessing large datasets to filter out noise.
    """

    def __init__(self, dataset, id=None, loader=None, strict=True, shuffler=None, index=None):
        assert loader is None or callable(loader)
        assert shuffler is None or callable(shuffler)        
        assert index is None or isinstance(index, (list, tuple))
        
        self._id = id
        self._ds = dataset if not isinstance(dataset, (list, set, tuple)) else tuple(dataset)  # force immutable (if possible)
        self._idx = list(range(len(self._ds)) if not index else index)
        self._loader = loader  # not serializable if lambda is provided
        self._shuffler = shuffler
        self._type = None
        
        assert not strict or index is None or (len(index)>0 and len(index)<=len(dataset) and max(index)<len(dataset) and min(index)>0)


    @classmethod
    def from_directory(cls, indir, filetype='json'):
        if filetype == 'json':
            return cls([x for f in vipy.util.findjson(indir) for x in to_iterable(vipy.load(f))])
        elif filetype in ['jpg']:
            return cls([vipy.image.Image(filename=f) for f in vipy.util.findjpeg(indir)])            
        else:
            raise ValueError('unsupported file type "%s"' % filetype)
    
    def __or__(self, other):
        assert isinstance(other, Dataset)
        return Union((self, other), id=self.id())

    
    def id(self, n=None, truncated=False, maxlen=80, suffix=None):
        """Set or return the dataset id, useful for showing the name/split of the dataset in the representation string"""
        if n is None and suffix is None:
            return (self._id[0:maxlex] + ' ... ') if truncated and self._id and len(self._id)>maxlen else self._id
        elif n is None and suffix is not None:
            self._id = self._id + suffix
        elif n is not None:
            self._id = n
        return self

    def index(self, index=None):
        """Update the index, useful for filtering of large datasets"""
        if index is not None:
            self._idx = index
            return self
        return self._idx
    
    def type(self):
        if self._type is None and len(self)>0:
            self._type = str(type(self[0]))  # peek at first element, cached
        return self._type
        
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
            return x
        elif isinstance(k, slice):
            X = [self._ds[k] for k in self._idx[k.start:k.stop:k.step]]
            X = [self._loader(x) for x in X] if self._loader is not None else X
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
    
    def tuple(self, mapper=None, flattener=to_iterable, reducer=None):
        """Return the dataset as a tuple, applying the optional mapper lambda on each element, applying optional flattener on sequences returned by mapper, and applying the optional reducer lambda on the final tuple, return a generator"""
        assert mapper is None or callable(mapper)
        assert flattener is None or callable(flattener)
        assert reducer is None or callable(reducer)        
        mapped = (mapper(x) if mapper else x for x in self)
        flattened = (y for x in mapped for y in flattener(x))
        reduced = reducer(flattened) if reducer else flattened
        return reduced

    def list(self, mapper=None, flattener=to_iterable, reducer=None):
        """Return a tuple as a list, loading into memory"""
        return list(self.tuple(mapper, flattener, reducer))

    def set(self, mapper):
        """Return the dataset as a set.  Mapper must be a lambda function that returns a hashable type"""
        return self.tuple(mapper=mapper, reducer=set)
        

    def frequency(self, f):
        """Frequency counts for which lamba returns the same value"""
        return vipy.util.countby(self.tuple(mapper=f))

    def count(self, f):
        """Counts for each element for which lamba returns true.  
        
        Args:
            f: [lambda] if provided, count the number of elements that return true.  

        Returns:
            A length of elements that satisfy f(v) = True [if f is not None]
        """
        return len(self.list(f, flattener=None, reducer=lambda X: [x for x in X if x is True]))

    def countby(self, f):
        return self.frequency(f)
    
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

    def takeby_asdict(self, f, n):
        """Group the dataset according to the callable f, take n from each group and return a dictionary"""
        return {k:v.takelist(n) for (k,v) in self.groupby(f).items()}
    
    def takeby(self, f, n):
        """Group the dataset according to the callable f, take n from each group and return a dataset"""
        return self.clone(shallow=True).index([i for (k,v) in self.groupby(f).items() for i in vipy.util.take(v._idx, n)])
    
    def takelist(self, n):
        """Take n elements and return list.  The elements are loaded and not cloned."""
        return self.take(n).list()

    def takeone(self):
        """Randomly take one element from the dataset and return a singleton"""
        return self[random.randint(0, len(self)-1)]

    def takeoneby(self, f):
        """Randomly take one element from the dataset and return a singleton if f(element) == True"""
        for k in shufflelist(self._idx):
            print(k)
            if f(self[k]):
                return self[k]
    
    def sample(self):
        return self.takeone()
    
    def take_fraction(self, p, inplace=False):
        """Randomly take a percentage of the dataset, returning a clone or in-place"""
        assert p>=0 and p<=1, "invalid fraction '%s'" % p
        return self.take(n=int(len(self)*p), inplace=inplace)

    def inverse_frequency(self, f):
        attributes = self.set(f)
        frequency = self.frequency(f)
        return {a:(1/len(attributes))*(len(self)/frequency[a]) for a in attributes}  # (normalized) inverse frequency weight
    
    def load(self):
        """Load the entire dataset into memory.  This is useful for creating in-memory datasets from lazy load datasets"""
        return Dataset(self.list(), id=self.id())
    
    def chunk(self, n):
        """Yield n chunks as dataset.  Last chunk will be ragged.  Batches are not loaded or preprocessed"""
        for (k,V) in enumerate(vipy.util.chunkgen(self._idx, n)):
            yield self.clone(shallow=True).index(V).id(('%s:%d' % (self.id(), k)) if self.id() else str(k))

    def batch(self, n):
        """Yield batches of size n as datasets.  Last batch will be ragged.  Batches are not loaded or preprocessed.  Batches have appended id equal to the zer-oindexed batch order"""
        for (k,V) in enumerate(vipy.util.chunkgenbysize(self._idx, n)):
            yield self.clone(shallow=True).index(V).id(('%s:%d' % (self.id(), k)) if self.id() else str(k))
            
    def minibatch(self, n, ragged=True):
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

        Returns:        
            Iterator over `vipy.dataset.Dataset` elements of length n.  Minibatches will be yielded loaded and preprocessed (processing done concurrently if vipy.parallel.executor() is initialized)

        ..note:: The distributed iterator appends the minibatch index to the minibatch.id().  
        ..note:: If there exists a vipy.parallel.exeuctor(), then loading and preprocessing will be performed concurrently

        """
        for b in vipy.parallel.map(lambda b: b.load(), self.batch(n)):  
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
    
    def pipeline(self, n, m, ragged=True, prepad=True, postpad=True):
        """Yield pipelined minibatches of size n with pipeline length m.

        A pipelined minibatch is a tuple (head, tail) such that (head, tail) are minibatches at different indexes in the dataset.  
        Head corresponds to the current minibatch and tail corresponds to the minibatch left shifted by (m-1) minibatches.

        This structure is useful for yielding datasets for pipelined training where head contains the minibatch that will complete pipeline training on this iteration, and tail contains the 
        next minibatch to be inserted into the pipeline on this iteration.
        
        ```python
        D = vipy.dataset.Dataset(...)
        for (head, tail) in D.pipeline(n, m, prepad=False, postpad=False):
            assert head == D[0:m]
            assert tail == D[n*(m-1): n*(m-1)+n]

        Args:
            n [int]: The size of each minibatch
            m [int]:  The pipeline length in minibatches
            ragged [bool]: If ragged=true, then the last chunk will be ragged with len(chunk)<n, else skipped
            prepad: If true, yield (head, tail) == (None, batch) when filling the pipeline
            postpad: If true, yield (head, tail) == (batch, None) when flushing the pipeline
        
        Returns:        
            Iterator over tuples (head,tail) of `vipy.dataset.Dataset` elements of length n where tail is left shifted by n*(m-1) elements. 
        
        .. note::  The distributed iterator is not order preserving over minibatches and yields minibatches as completed, however the tuple (head, tail) is order preserving within the pipeline
        .. note:: If there exists a vipy.parallel.executor(), then loading and preprocessing will be performed concurrently
        
        """
        pipeline = [] 
        for (k,b) in enumerate(self.minibatch(n, ragged=ragged)):  # not order preserving
            pipeline.append(b)  # order preserving within pipeline                        
            if k < m-1:
                if prepad:
                    yield( (None, b) )  
            else:
                yield( (pipeline.pop(0), b) )  # yield deque-like (minibatch, shifted minibatch) tuples
        for p in pipeline:
            if postpad:
                yield( (p, None) )


    def chunks(self, sizes):
        """Partition the dataset into chunks of size given by the tuple in partitions, and give the dataset suffix if provided"""
        assert sum(sizes) == len(self)

        i = 0
        datasets = []
        for n in sizes:
            datasets.append(self.clone(shallow=True).index(self._idx[i:i+n]))
            i += n
        return datasets

        
        
    def partition(self, trainfraction=0.9, valfraction=0.1, testfraction=0, trainsuffix=':train', valsuffix=':val', testsuffix=':test'):
        """Partition the dataset into the requested (train,val,test) fractions.  

        Args:
            trainfraction [float]: fraction of dataset for training set
            valfraction [float]: fraction of dataset for validation set
            testfraction [float]: fraction of dataset for test set
            trainsuffix: If not None, append this string the to trainset ID
            valsuffix: If not None, append this string the to valset ID
            testsuffix: If not None, append this string the to testset ID        
        
        Returns:        
            (trainset, valset, testset) such that trainset is the first trainfraction of the dataset.  

        .. note:: This does not permute the dataset.  To randomize split, shuffle dataset first

        """
        assert trainfraction >=0 and trainfraction <= 1, "invalid training set fraction '%f'" % trainfraction
        assert valfraction >=0 and valfraction <= 1, "invalid validation set fraction '%f'" % valfraction
        assert testfraction >=0 and testfraction <= 1, "invalid test set fraction '%f'" % testfraction
        assert abs(trainfraction + valfraction + testfraction - 1) < 1E-6, "fractions must sum to one"
        
        idx = self._idx
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
                
        return (trainset,valset,testset) if testfraction!=0 else (trainset, valset)

    def split(self, size):
        """Split the dataset into two datasets, one of length size, the other of length len(self)-size"""
        assert isinstance(size, int) and size>=0 and size<len(self)
        return self.partition(size/len(self), (len(self)-size)/len(self), 0, '', '', '')
        
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

    def zip(self, f, iter):
        return Dataset([f(im,i) for (im,i) in zip(self, iter)], id=self.id())  # triggers load into memory        
    
    def mapby_minibatch(self, f, n, ragged=True):
        return Dataset([f(b) for b in self.minibatch(n, ragged)], id=self.id())
    
    def sort(self, f):
        """Sort the dataset in-place using the sortkey lambda function f

        To perform a sort of the dataset using some property of the instance, such as the object category (e.g. for vipy.image objects) 

        ```python
        dataset.sort(lambda im: im.category())
        ```
        """
        self._idx = [self._idx[j] for (j,x) in sorted(zip(range(len(self)), self.tuple(f)), key=lambda x: x[1])]
        return self

    def randomize(self):
        """shuffle uniformly at random"""
        return Dataset.uniform_shuffler(self)
    
    @staticmethod
    def uniform_shuffler(D):
        random.shuffle(D._idx)        
        return D
    
    @staticmethod
    def identity_shuffler(D):
        """Shuffler that does nothing"""
        return D

    @staticmethod
    def chunk_shuffler(D, chunker, chunksize=64):
        """Split dataset into len(D)/chunksize non-overlapping chunks with some common property returned by chunker, shuffle chunk order and shuffle within chunks.  

           - If chunksize=1 then this is equivalent to uniform_shuffler
           - chunker must be a callable of some property that is used to group into chunks
            
        """
        assert callable(chunker)
        return D.randomize().sort(chunker).index([i for I in shufflelist([shufflelist(I) for I in vipy.util.chunkgenbysize(D._idx, chunksize)]) for i in I])
    
    
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
    
    def __init__(self, pagelist, loader, id=None, strict=True, index=None, cachesize=32, shuffler=None):        
        super().__init__(dataset=pagelist,
                         id=id,
                         loader=loader,
                         strict=False,
                         index=index if index else list(range(sum([p[0] for p in pagelist]))),
                         shuffler=shuffler)

        assert callable(loader), "page loader required"
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
            return x
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
        fields += ['union=%s' % str(tuple([d.id(truncated=True, maxlen=32) for d in self._ds]))]
        return str('<vipy.dataset.%s: %s>' % (self.__class__.__name__, ', '.join(fields)))
        
    def clone(self, shallow=False):
        """Return a copy of the dataset object"""
        D = super().clone(shallow=shallow)
        D._ds =  [d.clone(shallow=shallow) for d in D._ds]
        return D
    
    def datasets(self):
        """Return the dataset union elements, useful for generating unions of unions"""
        return list(self._ds)
    


def registry(name):
    (trainset, valset, testset) = (None, None, None)
    
    if name == 'mnist':
        import vipy.data.hf
        (trainset, testset) = vipy.data.hf.mnist()
        
    elif name == 'cifar10':
        import vipy.data.hf
        (trainset, testset) = vipy.data.hf.cifar10()
        
    else:
        raise ValueError('unknown dataset "%s"' % name)

    
    return (trainset, valset, testset)
